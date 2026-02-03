import os
import importlib
from io import BytesIO

import pandas as pd

st = importlib.import_module("streamlit")

from ma_pipeline import (
    WebExtractor,
    make_client,
    run_scoring,
    generate_full_description,
    generate_categories_from_keywords,
    generate_common_end_markets,
    tag_categories,
    tag_end_markets,
)

DEFAULT_CATEGORIES = []
DEFAULT_END_MARKETS = []


def _list_to_table(items, column_name):
    return pd.DataFrame({column_name: items}) if items else pd.DataFrame({column_name: []})


def _table_to_list(df, column_name):
    if df is None or column_name not in df.columns:
        return []
    values = [str(v).strip() for v in df[column_name].tolist()]
    return [v for v in values if v]


def _df_to_xlsx_bytes(df):
    output = BytesIO()
    df.to_excel(output, index=False)
    return output.getvalue()


st.set_page_config(page_title="M&A Target Scoring", layout="wide")
st.title("M&A Target Scoring")


with st.sidebar:
    st.header("API and runtime")
    api_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-5.2"))
    sector = st.text_input("Sector")

    scrape_mode = st.selectbox("Scraping mode", ["requests", "selenium"])
    chrome_path = st.text_input("Chrome path (selenium only)", value="")
    cache_path = st.text_input("Cache file (optional)", value="ma_cache.db")
    max_workers = st.number_input("Parallel fetch workers", min_value=1, max_value=12, value=4)

    delay_seconds = st.number_input("Delay between targets (sec)", min_value=0.0, value=3.0, step=0.5)
    score_threshold = st.number_input("Score threshold for enrichment", min_value=0, max_value=5, value=3)
    detect_subpages = st.checkbox("Auto-detect about/solutions/products", value=True)
    use_combined_enrichment = st.checkbox("Combined enrichment (faster)", value=True)

    st.subheader("Defaults")
    if "default_categories_df" not in st.session_state:
        st.session_state["default_categories_df"] = _list_to_table(DEFAULT_CATEGORIES, "Category")
    if "default_end_markets_df" not in st.session_state:
        st.session_state["default_end_markets_df"] = _list_to_table(DEFAULT_END_MARKETS, "End Market")

    default_categories_df = st.data_editor(
        st.session_state["default_categories_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="default_categories_editor",
    )
    default_end_markets_df = st.data_editor(
        st.session_state["default_end_markets_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="default_end_markets_editor",
    )

    st.session_state["default_categories_df"] = default_categories_df
    st.session_state["default_end_markets_df"] = default_end_markets_df
    apply_defaults = st.button("Apply defaults")


st.subheader("Client profile")

col1, col2 = st.columns(2)
with col1:
    client_companyname = st.text_input("Company name", value="")
    client_homepage = st.text_input("Homepage URL", value="")
    client_aboutuspage = st.text_input(
        "About URL",
        value="",
    )
with col2:
    client_productpage = st.text_input("Product URL", value="")
    client_solutionpage = st.text_input("Solutions URL", value="")

client_description_manual = st.text_area(
    "Client description (optional, overrides generated)",
    value="",
    height=140,
)

if "client_description" not in st.session_state:
    st.session_state["client_description"] = ""

if st.button("Generate client description"):
    if not api_key:
        st.error("Add your OpenAI API key to generate a description.")
    else:
        try:
            client = make_client(api_key)
            with WebExtractor(
                mode=scrape_mode,
                browser_executable_path=chrome_path,
                cache_path=cache_path or None,
                max_workers=int(max_workers),
            ) as extractor:
                homepage_text = extractor.extract_text(client_homepage)
                about_text = extractor.extract_text(client_aboutuspage)
                products_text = extractor.extract_text(client_productpage)
                solutions_text = extractor.extract_text(client_solutionpage)

                all_texts = [
                    ("About Us", about_text),
                    ("Solutions", solutions_text),
                    ("Products", products_text),
                    ("Homepage", homepage_text),
                ]

                description = generate_full_description(
                    client,
                    model,
                    company_name=client_companyname,
                    all_texts=all_texts,
                    commentary="",
                    headquarter="",
                    keywords="",
                    employee="",
                )
                st.session_state["client_description"] = description or ""
        except Exception as exc:
            st.error(f"Failed to generate client description: {exc}")

st.text_area(
    "Generated client description",
    value=st.session_state.get("client_description", ""),
    height=180,
    disabled=True,
)


st.subheader("Targets file")
uploaded_file = st.file_uploader("Upload targets file", type=["xlsx"])

if "df" not in st.session_state:
    st.session_state["df"] = None

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        if df.shape[1] < 2:
            st.error("Your file must have at least 2 columns: company and website.")
        else:
            df = df.iloc[:, :2].copy()
            df.columns = ["Company", "Homepage"]
            st.session_state["df"] = df
    except Exception as exc:
        st.error(f"Failed to read XLSX: {exc}")

df = st.session_state.get("df")
if df is not None:
    st.subheader("Upload preview")
    preview_rows = st.number_input("Rows to preview", min_value=1, max_value=200, value=10, step=1)
    st.dataframe(df.head(int(preview_rows)), use_container_width=True)
    colmap = {
        "company": "Company",
        "homepage": "Homepage",
        "headquarter": "Headquarter",
        "ftes": "FTEs",
        "about": "About Us",
        "solutions": "Solutions",
        "products": "Products",
    }

    st.subheader("Run scoring")
    start_index = st.number_input("Start index (0-based)", min_value=0, value=0, step=1)
    max_rows = st.number_input("Max rows (0 = all)", min_value=0, value=0, step=1)

    if st.button("Run scoring pipeline"):
        if not api_key:
            st.error("Add your OpenAI API key to run scoring.")
        else:
            client_description = client_description_manual.strip() or st.session_state.get("client_description", "")
            if not client_description:
                st.error("Provide a client description or generate one before running scoring.")
            else:
                progress = st.progress(0.0)
                status = st.empty()
                log_box = st.empty()
                log_lines = []

                def log_cb(message):
                    log_lines.append(message)
                    log_box.text("\n".join(log_lines[-200:]))

                def progress_cb(current, total, company):
                    pct = 0.0 if total == 0 else min(current / total, 1.0)
                    progress.progress(pct)
                    status.write(f"{current}/{total} {company}")

                try:
                    client = make_client(api_key)
                    max_rows_value = None if max_rows == 0 else int(max_rows)
                    with WebExtractor(
                        mode=scrape_mode,
                        browser_executable_path=chrome_path,
                        cache_path=cache_path or None,
                        max_workers=int(max_workers),
                    ) as extractor:
                        df = run_scoring(
                            df,
                            client,
                            model,
                            sector,
                            extractor,
                            client_description,
                            score_threshold=int(score_threshold),
                            delay_seconds=float(delay_seconds),
                            start_index=int(start_index),
                            max_rows=max_rows_value,
                            detect_subpages=detect_subpages,
                            use_combined_enrichment=use_combined_enrichment,
                            colmap=colmap,
                            log_cb=log_cb,
                            progress_cb=progress_cb,
                        )
                    st.session_state["df"] = df
                    st.success("Scoring completed.")
                except Exception as exc:
                    st.error(f"Scoring failed: {exc}")

    df = st.session_state.get("df")
    if df is not None:
        st.subheader("Scoring output preview")
        output_preview_rows = st.number_input("Rows to preview (scoring)", min_value=1, max_value=200, value=10, step=1)
        st.dataframe(df.head(int(output_preview_rows)), use_container_width=True)

        xlsx_bytes = _df_to_xlsx_bytes(df)
        st.download_button(
            "Download scoring XLSX",
            data=xlsx_bytes,
            file_name="scoring_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.subheader("Category tagging")
    if "categories_text" not in st.session_state:
        st.session_state["categories_text"] = ""
    if apply_defaults:
        defaults_list = _table_to_list(st.session_state["default_categories_df"], "Category")
        st.session_state["categories_text"] = "; ".join(defaults_list)

    if st.button("Generate categories from keywords"):
        if not api_key:
            st.error("Add your OpenAI API key to generate categories.")
        elif df is None or "Keywords" not in df.columns:
            st.error("Run scoring first to generate keywords.")
        else:
            client = make_client(api_key)
            categories_str = generate_categories_from_keywords(client, model, sector, df["Keywords"])
            if categories_str:
                st.session_state["categories_text"] = categories_str

    st.text_area("Categories (semicolon-separated)", key="categories_text", height=120)
    min_score_categories = st.number_input("Min score for category tagging", min_value=0, max_value=5, value=3)

    if st.button("Tag categories"):
        if not api_key:
            st.error("Add your OpenAI API key to tag categories.")
        elif df is None:
            st.error("Upload and score targets first.")
        else:
            progress = st.progress(0.0)
            status = st.empty()

            def progress_cb(current, total, company):
                pct = 0.0 if total == 0 else min(current / total, 1.0)
                progress.progress(pct)
                status.write(f"{current}/{total} {company}")

            client = make_client(api_key)
            categories = [c.strip() for c in st.session_state["categories_text"].replace("\n", ";").split(";") if c.strip()]
            with WebExtractor(
                mode=scrape_mode,
                browser_executable_path=chrome_path,
                cache_path=cache_path or None,
                max_workers=int(max_workers),
            ) as extractor:
                df = tag_categories(
                    df,
                    client,
                    model,
                    sector,
                    extractor,
                    categories,
                    min_score=int(min_score_categories),
                    delay_seconds=float(delay_seconds),
                    colmap=colmap,
                    progress_cb=progress_cb,
                )
            st.session_state["df"] = df
            st.success("Category tagging completed.")

    st.subheader("End market tagging")
    if "end_markets_text" not in st.session_state:
        st.session_state["end_markets_text"] = ""
    if apply_defaults:
        defaults_list = _table_to_list(st.session_state["default_end_markets_df"], "End Market")
        st.session_state["end_markets_text"] = "; ".join(defaults_list)

    if st.button("Generate end markets from extracted end markets"):
        if not api_key:
            st.error("Add your OpenAI API key to generate end markets.")
        elif df is None or "End Markets" not in df.columns:
            st.error("Run scoring first to generate end markets.")
        else:
            client = make_client(api_key)
            end_markets_str = generate_common_end_markets(client, model, df["End Markets"])
            if end_markets_str:
                st.session_state["end_markets_text"] = end_markets_str

    st.text_area("End markets (semicolon-separated)", key="end_markets_text", height=120)
    min_score_end_markets = st.number_input("Min score for end market tagging", min_value=0, max_value=5, value=0)

    if st.button("Tag end markets"):
        if not api_key:
            st.error("Add your OpenAI API key to tag end markets.")
        elif df is None:
            st.error("Upload and score targets first.")
        else:
            progress = st.progress(0.0)
            status = st.empty()

            def progress_cb(current, total, company):
                pct = 0.0 if total == 0 else min(current / total, 1.0)
                progress.progress(pct)
                status.write(f"{current}/{total} {company}")

            client = make_client(api_key)
            end_markets = [c.strip() for c in st.session_state["end_markets_text"].replace("\n", ";").split(";") if c.strip()]
            with WebExtractor(
                mode=scrape_mode,
                browser_executable_path=chrome_path,
                cache_path=cache_path or None,
                max_workers=int(max_workers),
            ) as extractor:
                df = tag_end_markets(
                    df,
                    client,
                    model,
                    extractor,
                    end_markets,
                    min_score=int(min_score_end_markets),
                    delay_seconds=float(delay_seconds),
                    colmap=colmap,
                    progress_cb=progress_cb,
                )
            st.session_state["df"] = df
            st.success("End market tagging completed.")

    df = st.session_state.get("df")
    if df is not None:
        st.subheader("Final output preview")
        final_preview_rows = st.number_input("Rows to preview (final)", min_value=1, max_value=200, value=10, step=1)
        st.dataframe(df.head(int(final_preview_rows)), use_container_width=True)

        xlsx_bytes = _df_to_xlsx_bytes(df)
        st.download_button(
            "Download full XLSX",
            data=xlsx_bytes,
            file_name="ma_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
