import time
import re
import shelve
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from urllib.parse import urljoin
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MIN_TEXT_LENGTH = 200
JINA_TEXT_BASE = "https://r.jina.ai/http://"


def _safe_str(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_url(url):
    if url is None:
        return ""
    url = str(url).strip()
    if not url:
        return ""
    if not url.startswith("http"):
        url = "https://" + url
    return url


def _combine_texts(all_texts, max_chars):
    combined = ""
    for label, content in all_texts:
        text = _safe_str(content)
        if text:
            remaining = max_chars - len(combined)
            if remaining <= 0:
                break
            combined += f"\n\n--- {label} ---\n{text[:remaining]}"
    return combined.strip()


def _call_chat(client, model, prompt, temperature=0.2, max_tokens=200):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _extract_with_trafilatura(html):
    try:
        trafilatura = importlib.import_module("trafilatura")
        return trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except Exception:
        return None


def _fetch_with_jina(url):
    try:
        if url.startswith("https://"):
            jina_url = JINA_TEXT_BASE + url[len("https://") :]
        elif url.startswith("http://"):
            jina_url = JINA_TEXT_BASE + url[len("http://") :]
        else:
            jina_url = JINA_TEXT_BASE + url
        response = requests.get(jina_url, timeout=20)
        response.raise_for_status()
        return response.text
    except Exception:
        return None


class WebExtractor:
    def __init__(
        self,
        mode="requests",
        headless=True,
        browser_executable_path=None,
        page_wait_seconds=10,
        cache_path=None,
        max_workers=4,
        use_jina_fallback=False,
    ):
        self.mode = mode
        self.headless = headless
        self.browser_executable_path = browser_executable_path
        self.page_wait_seconds = page_wait_seconds
        self.max_workers = max_workers
        self.use_jina_fallback = use_jina_fallback
        self.driver: Any = None
        self._by: Any = None
        self._wait: Any = None
        self._ec: Any = None
        self._session = requests.Session()
        self._cache_path = cache_path
        self._cache = None
        if self._cache_path:
            self._cache = shelve.open(self._cache_path)
        if self.mode == "selenium":
            self._init_driver()

    def _init_driver(self):
        try:
            uc = importlib.import_module("undetected_chromedriver")
            selenium_webdriver = importlib.import_module("selenium.webdriver")
            selenium_by = importlib.import_module("selenium.webdriver.common.by")
            selenium_wait = importlib.import_module("selenium.webdriver.support.ui")
            selenium_ec = importlib.import_module("selenium.webdriver.support.expected_conditions")
        except Exception as exc:
            raise RuntimeError("Selenium dependencies not available") from exc

        options = uc.ChromeOptions()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        if self.browser_executable_path:
            self.driver = uc.Chrome(options=options, browser_executable_path=self.browser_executable_path)
        else:
            self.driver = uc.Chrome(options=options)

        self._by = selenium_by.By
        self._wait = selenium_wait.WebDriverWait
        self._ec = selenium_ec

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
        if self._cache is not None:
            self._cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _get_page_source(self, url):
        url = _normalize_url(url)
        if not url:
            return ""

        if self.mode == "selenium":
            if self.driver is None or self._wait is None or self._ec is None or self._by is None:
                raise RuntimeError("Selenium driver not initialized")
            self.driver.get(url)
            self._wait(self.driver, self.page_wait_seconds).until(
                self._ec.presence_of_element_located((self._by.TAG_NAME, "body"))
            )
            return self.driver.page_source

        headers = {"User-Agent": DEFAULT_USER_AGENT}
        try:
            response = self._session.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            if url.startswith("https://"):
                fallback_url = "http://" + url[len("https://") :]
                response = self._session.get(fallback_url, headers=headers, timeout=20)
                response.raise_for_status()
                return response.text
            raise exc

    def extract_text(self, url):
        url = _normalize_url(url)
        if not url:
            return ""
        cache_key = f"text::{url}"
        if self._cache is not None and cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached:
                return cached

        try:
            page_source = self._get_page_source(url)
        except Exception:
            page_source = ""

        if not page_source and self.use_jina_fallback:
            jina_text = _fetch_with_jina(url)
            if jina_text:
                jina_text = re.sub(r"\s+", " ", jina_text.strip())
                if self._cache is not None and jina_text:
                    self._cache[cache_key] = jina_text
                return jina_text

        if not page_source:
            return ""

        soup = BeautifulSoup(page_source, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)

        if len(text) < MIN_TEXT_LENGTH:
            tf_text = _extract_with_trafilatura(page_source)
            if tf_text:
                tf_text = re.sub(r"\s+", " ", tf_text.strip())
                if len(tf_text) > len(text):
                    text = tf_text

        if len(text) < MIN_TEXT_LENGTH and self.use_jina_fallback:
            jina_text = _fetch_with_jina(url)
            if jina_text:
                jina_text = re.sub(r"\s+", " ", jina_text.strip())
                if len(jina_text) > len(text):
                    text = jina_text

        if self._cache is not None and text:
            self._cache[cache_key] = text
        return text

    def extract_texts(self, urls):
        unique_urls = [_normalize_url(u) for u in urls]
        unique_urls = [u for u in unique_urls if u]
        if not unique_urls:
            return {}

        results = {}
        if self.mode == "selenium" or self.max_workers <= 1:
            for url in unique_urls:
                results[url] = self.extract_text(url)
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(self.extract_text, url): url for url in unique_urls}
            for future in as_completed(future_map):
                url = future_map[future]
                try:
                    results[url] = future.result()
                except Exception:
                    results[url] = ""
        return results

    def get_soup(self, url):
        try:
            page_source = self._get_page_source(url)
        except Exception:
            return None
        if not page_source:
            return None
        soup = BeautifulSoup(page_source, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup


def find_navigation_block(soup):
    nav = soup.find("nav") or soup.find("header")
    if nav:
        return nav
    nav_candidates = soup.find_all(
        "div", attrs={"class": re.compile(r".*(menu|nav|header).*", re.IGNORECASE)}
    )
    if nav_candidates:
        return nav_candidates[0]
    nav_candidates = soup.find_all(
        "div", attrs={"id": re.compile(r".*(menu|nav|header).*", re.IGNORECASE)}
    )
    if nav_candidates:
        return nav_candidates[0]
    return None


def find_sector_pages_with_gpt(client, model, extractor, homepage_url, sector):
    try:
        soup = extractor.get_soup(homepage_url)
        if soup is None:
            return {"About Us": None, "Solutions": None, "Products": None, "status": "navigation_not_found"}

        nav_block = find_navigation_block(soup)
        if nav_block is None:
            return {"About Us": None, "Solutions": None, "Products": None, "status": "navigation_not_found"}

        html_fragment = str(nav_block)[:8000]
        prompt = (
            f"You are an M&A analyst specializing in the {sector} sector.\n"
            "You will receive the HTML code from a company's homepage (mostly navigation links).\n"
            "Your task is to identify the internal URLs that are the most relevant for analyzing the company's activity.\n\n"
            "Specifically, return:\n"
            "- The most relevant 'About Us' page (if any)\n"
            "- The best page describing solutions/services\n"
            "- The most representative product page (flagship or legacy product)\n\n"
            "Format your response exactly like this:\n"
            "About Us URL: ...\n"
            "Solutions URL: ...\n"
            "Product URL: ...\n\n"
            "If any of them cannot be found, write 'None'.\n\n"
            f"--- HTML ---\n{html_fragment}"
        )

        answer = _call_chat(client, model, prompt, temperature=0, max_tokens=500)
        result: dict[str, Any] = {"About Us": None, "Solutions": None, "Products": None}
        patterns = {
            "About Us": r"About Us URL:\s*(\S+)",
            "Solutions": r"Solutions URL:\s*(\S+)",
            "Products": r"Product URL:\s*(\S+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, answer)
            if match:
                url = match.group(1).strip()
                if url.lower() == "none":
                    continue
                result[key] = urljoin(homepage_url, url) if url.startswith("/") else url

        result["status"] = "ok"
        return result
    except Exception:
        return {"About Us": None, "Solutions": None, "Products": None, "status": "exception"}


def generate_full_description(
    client,
    model,
    company_name,
    all_texts,
    commentary,
    headquarter,
    keywords,
    employee,
    raise_on_error=False,
):
    combined_text = _combine_texts(all_texts, max_chars=6000)
    if not combined_text:
        return None

    prompt = (
        "You are an M&A analyst researching acquisition targets.\n"
        "Based on the structured content below, write a comprehensive description of the company in English. "
        "Include what the company is, what they do, sectors of activity, key products or services, "
        "location, approximate size (employees), and strategic positioning.\n\n"
        f"Company name: {company_name}\n"
        f"{combined_text}\n"
    )

    if _safe_str(commentary).strip():
        prompt += f"\n--- Additional commentaries ---\n{_safe_str(commentary).strip()}"
    if _safe_str(headquarter).strip():
        prompt += f"\nHeadquarter: {_safe_str(headquarter).strip()}"
    if _safe_str(employee).strip():
        prompt += f"\nEmployees: {_safe_str(employee).strip()}"
    if _safe_str(keywords).strip():
        prompt += f"\nRelevant keywords: {_safe_str(keywords).strip()}"

    try:
        return _call_chat(client, model, prompt, temperature=0.2, max_tokens=500)
    except Exception:
        if raise_on_error:
            raise
        return None


def parse_gpt_response(content):
    score_match = re.search(r"Score\s*:\s*([0-5])", content, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else None
    bullets = re.findall(r"^\s*[-\u2022]\s*(.+)", content, re.MULTILINE)
    justifications = []
    for bullet in bullets[:3]:
        cleaned = re.sub(
            r"^justification\s+bullet\s+point\s+\d+:\s*",
            "",
            bullet.strip(),
            flags=re.IGNORECASE,
        )
        justifications.append(cleaned)
    while len(justifications) < 3:
        justifications.append(None)
    return score, justifications[0], justifications[1], justifications[2]


def generate_score(client, model, company_name, all_texts, headquarter, ftes, description_client):
    combined_text = _combine_texts(all_texts, max_chars=7000)
    if not combined_text:
        return None, None, None, None

    prompt = (
        "You are an M&A analyst evaluating acquisition targets.\n"
        "Your task is to evaluate Company B as a potential acquisition target for Company A, "
        "based solely on the provided descriptions and business information.\n\n"
        f"Company A (client):\n{description_client}\n"
        f"Company B (target):\n{combined_text}\n"
    )
    if _safe_str(headquarter).strip():
        prompt += f"\nHeadquarter: {_safe_str(headquarter).strip()}"
    if _safe_str(ftes).strip():
        prompt += f"\nFTEs: {_safe_str(ftes).strip()}"

    prompt += (
        "\n\n--- Scoring Criteria ---\n"
        "Evaluate the relevance of Company B as a potential M&A target for Company A based on:\n"
        "- Product and service similarity (overlapping offerings)\n"
        "- Complementarity of offerings\n"
        "- Business model compatibility\n\n"
        "Respond strictly in this format:\n"
        "Score: X (where X is an integer from 0 to 5)\n"
        "- Justification bullet point 1 (1 to 2 concise sentences)\n"
        "- Justification bullet point 2 (same format)\n"
        "- Justification bullet point 3 (same format)\n\n"
        "Scoring scale:\n"
        "0 = Not relevant at all\n"
        "1 = Slightly relevant\n"
        "2 = Somewhat relevant\n"
        "3 = Moderately relevant\n"
        "4 = Very relevant\n"
        "5 = Highly relevant M&A target"
    )

    try:
        content = _call_chat(client, model, prompt, temperature=0, max_tokens=500)
        return parse_gpt_response(content)
    except Exception:
        return None, None, None, None


def generate_description(client, model, company_name, partial_texts, headquarter=None, ftes=None):
    combined_text = _combine_texts(partial_texts, max_chars=6000)
    if not combined_text:
        return None

    prompt = (
        "You are a business analyst specializing in M&A.\n"
        "Your goal is to summarize what this company does, based on available online content and metadata.\n\n"
        f"Company name: {company_name}\n"
        f"Website content:\n{combined_text}\n"
    )

    if _safe_str(headquarter).strip():
        prompt += f"\nHeadquarter: {_safe_str(headquarter).strip()}"
    if _safe_str(ftes).strip():
        prompt += f"\nFTEs: {_safe_str(ftes).strip()}"

    prompt += (
        "\n\nWrite a concise 2 to 3 sentence description in English of what the company does and its main business activities. "
        "If information is unclear, provide your best synthesis based on the clues available."
    )

    try:
        return _call_chat(client, model, prompt, temperature=0.2, max_tokens=200)
    except Exception:
        return None


def generate_keywords(client, model, company_name, partial_texts, headquarter=None, ftes=None):
    combined_text = _combine_texts(partial_texts, max_chars=6000)
    if not combined_text:
        return None

    prompt = (
        "You are a business analyst working on market mapping for M&A.\n"
        "Based on the company's homepage and about page, identify relevant keywords summarizing its activity.\n\n"
        f"Company name: {company_name}\n"
        f"Website content:\n{combined_text}\n"
    )

    if _safe_str(headquarter).strip():
        prompt += f"\nHeadquarter: {_safe_str(headquarter).strip()}"
    if _safe_str(ftes).strip():
        prompt += f"\nFTEs: {_safe_str(ftes).strip()}"

    prompt += (
        "\n\nReturn a list of 5 to 10 relevant business keywords (products, technologies, markets, expertise). "
        "Separate them using a semicolon (;) and do not add explanations."
    )

    try:
        return _call_chat(client, model, prompt, temperature=0.2, max_tokens=150)
    except Exception:
        return None


def generate_categories_from_keywords(client, model, sector, all_keywords):
    keyword_list = []
    for kw in all_keywords:
        if pd.notna(kw):
            parts = [k.strip().lower() for k in str(kw).split(";") if k.strip()]
            keyword_list.extend(parts)

    if not keyword_list:
        return None

    keyword_counts = Counter(keyword_list)
    sorted_keywords = [f"{kw} ({count})" for kw, count in keyword_counts.most_common()]

    prompt = (
        f"You are analyzing companies in the '{sector}' sector.\n"
        "The list below contains recurring technical or operational keywords extracted from company descriptions, along with their frequency in parentheses.\n\n"
        "Your task is to identify 10 to 15 precise and meaningful technical or functional categories that best organize these keywords into coherent groups.\n\n"
        "Guidelines:\n"
        "- Assume all companies are part of the sector; do not include the sector name as a category.\n"
        "- Each category must represent a distinct technical, operational, or service-oriented theme.\n"
        "- Do not use vague categories like 'consulting' or 'services' unless qualified.\n"
        "- Prefer industry-specific language.\n"
        "- Try to make categories mutually exclusive and collectively exhaustive.\n"
        "- Each category should encompass several related keywords.\n\n"
        "Output format:\n"
        "Return a single-line list of category names, separated by semicolons (;), and nothing else.\n\n"
        "Keywords:\n" + "\n".join(sorted_keywords)
    )

    try:
        return _call_chat(client, model, prompt, temperature=0.2, max_tokens=700)
    except Exception:
        return None


def categorize(client, model, sector, company_name, partial_texts, keywords, description, categories):
    combined_text = _combine_texts(partial_texts, max_chars=4000)
    if not combined_text:
        return []

    prompt = (
        f"You are an M&A analyst in the '{sector}' sector.\n"
        "Based on the following company content, select the most relevant categories from this list:\n"
        "- " + "\n- ".join(categories) + "\n\n"
        f"Company: {company_name}\n"
        f"Keywords: {_safe_str(keywords)}\n"
        f"Description: {_safe_str(description)}\n"
        f"Website content:\n{combined_text}\n\n"
        "Return only the matching categories, as a semicolon-separated list."
    )

    try:
        raw_output = _call_chat(client, model, prompt, temperature=0.2, max_tokens=200)
        category_map = {c.lower(): c for c in categories}
        matched = []
        for cat in raw_output.split(";"):
            key = cat.strip().lower()
            if key in category_map:
                matched.append(category_map[key])
        return matched
    except Exception:
        return []


def generate_end_market(client, model, company_name, partial_texts, headquarter=None, ftes=None):
    combined_text = _combine_texts(partial_texts, max_chars=6000)
    if not combined_text:
        return None

    prompt = (
        "You are a business analyst working on sector mapping for M&A sourcing.\n"
        "Your task is to identify the key end markets in which a company operates, based on its website content.\n\n"
        "An end market refers to a concrete domain or type of physical asset the company serves.\n"
        "Avoid vague answers like 'industry' or 'technology'. Focus on specific asset types or industries.\n\n"
        f"Company name: {company_name}\n"
        f"Website content:\n{combined_text}\n"
    )

    if _safe_str(headquarter).strip():
        prompt += f"\nHeadquarter: {_safe_str(headquarter).strip()}"
    if _safe_str(ftes).strip():
        prompt += f"\nFTEs: {_safe_str(ftes).strip()}"

    prompt += (
        "\n\nReturn a list of 3 to 8 relevant end markets for this company. "
        "Use only short labels, separated by a semicolon (;). "
        "Do not provide explanations or bullet points."
    )

    try:
        return _call_chat(client, model, prompt, temperature=0.2, max_tokens=100)
    except Exception:
        return None


def generate_common_end_markets(client, model, all_end_markets):
    keyword_list = []
    for kw in all_end_markets:
        if pd.notna(kw):
            parts = [k.strip().lower() for k in str(kw).split(";") if k.strip()]
            keyword_list.extend(parts)

    if not keyword_list:
        return None

    keyword_counts = Counter(keyword_list)
    sorted_keywords = [f"{kw} ({count})" for kw, count in keyword_counts.most_common()]

    prompt = (
        "You are an M&A analyst specializing in industrial and infrastructure sectors.\n"
        "You are reviewing keywords extracted from company websites to identify the most relevant end markets.\n\n"
        "Your task is to return a concise list of 6 to 10 distinct end markets, based on the recurring patterns below.\n\n"
        "Guidelines:\n"
        "- Think in terms of physical domains or industry applications.\n"
        "- Do not use vague categories like 'technology' or 'systems'.\n"
        "- Focus on end-use contexts or major types of assets.\n"
        "- Use short, precise labels. Avoid explanations.\n\n"
        "Output format: Return a semicolon-separated list of end market names, all on a single line.\n\n"
        "Extracted keywords with frequency:\n" + "\n".join(sorted_keywords)
    )

    try:
        return _call_chat(client, model, prompt, temperature=0.2, max_tokens=300)
    except Exception:
        return None


def generate_enrichment(client, model, company_name, partial_texts, headquarter=None, ftes=None):
    combined_text = _combine_texts(partial_texts, max_chars=6000)
    if not combined_text:
        return None, None, None

    prompt = (
        "You are a business analyst specializing in M&A.\n"
        "Summarize what this company does, then extract keywords and end markets.\n\n"
        f"Company name: {company_name}\n"
        f"Website content:\n{combined_text}\n"
    )

    if _safe_str(headquarter).strip():
        prompt += f"\nHeadquarter: {_safe_str(headquarter).strip()}"
    if _safe_str(ftes).strip():
        prompt += f"\nFTEs: {_safe_str(ftes).strip()}"

    prompt += (
        "\n\nReturn the result in this exact format (single line per item):\n"
        "Description: <2 to 3 sentences>\n"
        "Keywords: <5 to 10 keywords separated by semicolons>\n"
        "End Markets: <3 to 8 end markets separated by semicolons>"
    )

    try:
        content = _call_chat(client, model, prompt, temperature=0.2, max_tokens=250)
    except Exception:
        return None, None, None

    match = re.search(
        r"Description:\s*(.*?)\nKeywords:\s*(.*?)\nEnd Markets:\s*(.*)",
        content,
        flags=re.DOTALL,
    )
    if not match:
        return None, None, None

    description = match.group(1).strip()
    keywords = match.group(2).strip()
    end_markets = match.group(3).strip()
    return description, keywords, end_markets


def categorize_end_markets(client, model, company_name, partial_texts, keywords, description, categories):
    combined_text = _combine_texts(partial_texts, max_chars=4000)
    if not combined_text:
        return []

    prompt = (
        "You are an M&A analyst classifying companies into end markets based on their activity.\n"
        "An end market is a specific domain or type of physical infrastructure the company targets.\n\n"
        "Your task is to select the most relevant end markets from the list below.\n\n"
        "List of possible end markets:\n- " + "\n- ".join(categories) + "\n\n"
        f"Company: {company_name}\n"
        f"Keywords: {_safe_str(keywords)}\n"
        f"Description: {_safe_str(description)}\n"
        f"Website content:\n{combined_text}\n\n"
        "Return only the relevant end markets as a semicolon-separated list."
    )

    try:
        raw_output = _call_chat(client, model, prompt, temperature=0.2, max_tokens=200)
        category_map = {c.lower(): c for c in categories}
        matched = []
        for cat in raw_output.split(";"):
            key = cat.strip().lower()
            if key in category_map:
                matched.append(category_map[key])
        return matched
    except Exception:
        return []


def ensure_text_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)


def run_scoring(
    df,
    client,
    model,
    sector,
    extractor,
    client_description,
    score_threshold=3,
    delay_seconds=3.0,
    start_index=0,
    max_rows=None,
    detect_subpages=True,
    use_combined_enrichment=True,
    colmap=None,
    log_cb=None,
    progress_cb=None,
):
    colmap = colmap or {}
    company_col = colmap.get("company", "Company")
    homepage_col = colmap.get("homepage", "Homepage")
    headquarter_col = colmap.get("headquarter", "Headquarter")
    ftes_col = colmap.get("ftes", "FTEs")
    about_col = colmap.get("about", "About Us")
    solutions_col = colmap.get("solutions", "Solutions")
    products_col = colmap.get("products", "Products")

    ensure_text_columns(
        df,
        [
            about_col,
            solutions_col,
            products_col,
            "Justification 1",
            "Justification 2",
            "Justification 3",
            "Description",
            "Keywords",
            "End Markets",
        ],
    )

    total_rows = len(df)
    if max_rows is not None:
        total_rows = min(total_rows, start_index + max_rows)

    if log_cb:
        log_cb(f"Starting from row {start_index + 1} of {len(df)}")

    for idx in range(start_index, total_rows):
        company = _safe_str(df.at[idx, company_col])
        homepage = _normalize_url(_safe_str(df.at[idx, homepage_col]))
        if progress_cb:
            progress_cb(idx - start_index + 1, total_rows - start_index, company)

        if not homepage:
            if log_cb:
                log_cb(f"Skipping: missing or invalid homepage for {company}")
            continue

        df.at[idx, homepage_col] = homepage

        if detect_subpages:
            found_links = find_sector_pages_with_gpt(client, model, extractor, homepage, sector)
            if found_links.get("status") == "exception":
                if log_cb:
                    log_cb(f"Skipping: subpage detection failed for {homepage}")
                continue

            df.at[idx, about_col] = _safe_str(found_links.get("About Us"))
            df.at[idx, solutions_col] = _safe_str(found_links.get("Solutions"))
            df.at[idx, products_col] = _safe_str(found_links.get("Products"))

        homepage_url = df.at[idx, homepage_col]
        about_url = df.at[idx, about_col]
        solutions_url = df.at[idx, solutions_col]
        products_url = df.at[idx, products_col]

        texts = extractor.extract_texts([homepage_url, about_url, solutions_url, products_url])
        homepage_text = texts.get(_normalize_url(homepage_url), "")
        about_text = texts.get(_normalize_url(about_url), "")
        solutions_text = texts.get(_normalize_url(solutions_url), "")
        products_text = texts.get(_normalize_url(products_url), "")

        text_lengths = {
            "homepage": len(homepage_text or ""),
            "about": len(about_text or ""),
            "solutions": len(solutions_text or ""),
            "products": len(products_text or ""),
        }
        if log_cb and sum(text_lengths.values()) == 0:
            log_cb(
                "No text extracted for "
                f"{company}. Homepage: {homepage_url or 'missing'}. "
                "Check URLs or try a different scraping mode."
            )

        all_texts = [
            ("About Us", about_text),
            ("Solutions", solutions_text),
            ("Products", products_text),
            ("Homepage", homepage_text),
        ]

        partial_texts = [
            ("About Us", about_text),
            ("Homepage", homepage_text),
        ]

        score, just1, just2, just3 = generate_score(
            client,
            model,
            company_name=company,
            all_texts=all_texts,
            headquarter=df.at[idx, headquarter_col] if headquarter_col in df.columns else "",
            ftes=df.at[idx, ftes_col] if ftes_col in df.columns else "",
            description_client=client_description,
        )

        if score is not None:
            df.at[idx, "Score"] = score
            df.at[idx, "Justification 1"] = just1
            df.at[idx, "Justification 2"] = just2
            df.at[idx, "Justification 3"] = just3

            if score >= score_threshold:
                hq_value = df.at[idx, headquarter_col] if headquarter_col in df.columns else ""
                ftes_value = df.at[idx, ftes_col] if ftes_col in df.columns else ""

                if use_combined_enrichment:
                    description, keywords, end_markets = generate_enrichment(
                        client,
                        model,
                        company_name=company,
                        partial_texts=partial_texts,
                        headquarter=hq_value,
                        ftes=ftes_value,
                    )
                else:
                    description = generate_description(
                        client,
                        model,
                        company_name=company,
                        partial_texts=partial_texts,
                        headquarter=hq_value,
                        ftes=ftes_value,
                    )
                    keywords = generate_keywords(
                        client,
                        model,
                        company_name=company,
                        partial_texts=partial_texts,
                        headquarter=hq_value,
                        ftes=ftes_value,
                    )
                    end_markets = generate_end_market(
                        client,
                        model,
                        company_name=company,
                        partial_texts=partial_texts,
                        headquarter=hq_value,
                        ftes=ftes_value,
                    )

                if description:
                    df.at[idx, "Description"] = description
                if keywords:
                    df.at[idx, "Keywords"] = keywords
                if end_markets:
                    df.at[idx, "End Markets"] = end_markets

        if delay_seconds:
            time.sleep(delay_seconds)

    return df


def tag_categories(
    df,
    client,
    model,
    sector,
    extractor,
    categories,
    min_score=3,
    delay_seconds=3.0,
    colmap=None,
    log_cb=None,
    progress_cb=None,
):
    colmap = colmap or {}
    company_col = colmap.get("company", "Company")
    homepage_col = colmap.get("homepage", "Homepage")
    about_col = colmap.get("about", "About Us")

    ensure_text_columns(df, [company_col, homepage_col, about_col, "Score", "Keywords", "Description"])

    categories = [c.strip() for c in categories if c.strip()]
    for cat in categories:
        if cat not in df.columns:
            df[cat] = ""

    total_rows = len(df)
    for idx in range(total_rows):
        company = _safe_str(df.at[idx, company_col])
        if progress_cb:
            progress_cb(idx + 1, total_rows, company)

        score_num = _to_float(df.at[idx, "Score"])
        if score_num is None or pd.isna(score_num) or score_num < min_score:
            continue

        homepage_text = extractor.extract_text(df.at[idx, homepage_col])
        about_text = extractor.extract_text(df.at[idx, about_col])
        partial_texts = [("About Us", about_text), ("Homepage", homepage_text)]

        matched = categorize(
            client,
            model,
            sector,
            company_name=company,
            partial_texts=partial_texts,
            keywords=df.at[idx, "Keywords"],
            description=df.at[idx, "Description"],
            categories=categories,
        )

        for cat in matched:
            df.at[idx, cat] = "P"

        if delay_seconds:
            time.sleep(delay_seconds)

    return df


def tag_end_markets(
    df,
    client,
    model,
    extractor,
    end_markets,
    min_score=0,
    delay_seconds=3.0,
    colmap=None,
    log_cb=None,
    progress_cb=None,
):
    colmap = colmap or {}
    company_col = colmap.get("company", "Company")
    homepage_col = colmap.get("homepage", "Homepage")
    about_col = colmap.get("about", "About Us")

    ensure_text_columns(df, [company_col, homepage_col, about_col, "Score", "End Markets", "Description"])

    end_markets = [c.strip() for c in end_markets if c.strip()]
    for cat in end_markets:
        if cat not in df.columns:
            df[cat] = ""

    total_rows = len(df)
    for idx in range(total_rows):
        company = _safe_str(df.at[idx, company_col])
        if progress_cb:
            progress_cb(idx + 1, total_rows, company)

        score_num = _to_float(df.at[idx, "Score"])
        if score_num is None or pd.isna(score_num) or score_num < min_score:
            continue

        homepage_text = extractor.extract_text(df.at[idx, homepage_col])
        about_text = extractor.extract_text(df.at[idx, about_col])
        partial_texts = [("About Us", about_text), ("Homepage", homepage_text)]

        matched = categorize_end_markets(
            client,
            model,
            company_name=company,
            partial_texts=partial_texts,
            keywords=df.at[idx, "End Markets"],
            description=df.at[idx, "Description"],
            categories=end_markets,
        )

        for cat in matched:
            df.at[idx, cat] = "P"

        if delay_seconds:
            time.sleep(delay_seconds)

    return df


def make_client(api_key):
    openai_module = importlib.import_module("openai")
    return openai_module.OpenAI(api_key=api_key)
