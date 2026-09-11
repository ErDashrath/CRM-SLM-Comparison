"""
Dataset Browser -- a second page (Streamlit's standard ui/pages/ convention,
auto-appears in the sidebar nav alongside app.py) for browsing the full
mock CRM dataset every model variant in this project is trained and
evaluated against: per-account profile, opportunity/deal facts, every
email, every call transcript, and the company-wide playbooks/catalog --
all selectable, all rendered in full, not summarized or truncated.

Reuses common/context.py::assemble_context() (the same function training,
eval, and the comparison page all read through) rather than re-implementing
CRM-folder parsing, so this page always reflects exactly what the models
actually see.

Run via the same entrypoint as app.py -- Streamlit auto-discovers this file:
    /home/dsp-at-magna/Magna/venv-gpu/bin/python -m streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_PAGE_DIR = Path(__file__).resolve().parent
_APP_DIR = _PAGE_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from common.context import assemble_context, list_account_ids

st.set_page_config(page_title="CRM Dataset Browser", layout="wide")

HEALTH_BADGE = {"healthy": "🟢 Healthy", "at_risk": "🔴 At risk", "watch": "🟡 Watch"}
SEVERITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}

st.title("CRM Dataset Browser")
st.caption(
    "The full mock CRM dataset this project's models are trained and evaluated against -- "
    "every account, email, call transcript, and company-wide playbook, selectable and shown in full."
)

account_ids = list_account_ids()
account_id = st.selectbox("Account", account_ids, format_func=lambda a: a.replace("_", " ").title())

hits = assemble_context(account_id)
by_type: dict[str, list[dict]] = defaultdict(list)
for h in hits:
    by_type[h["doc_type"]].append(h)

account_data = json.loads(by_type["account"][0]["text"]) if by_type.get("account") else None
opp_data = json.loads(by_type["opportunity"][0]["text"]) if by_type.get("opportunity") else None

# --- Account + opportunity overview -----------------------------------------

if account_data:
    st.subheader(account_data["name"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Industry", account_data["industry"])
    col2.metric("Segment", account_data["segment"])
    col3.metric("Health", HEALTH_BADGE.get(account_data["account_health"], account_data["account_health"]))
    col4.metric("Account Owner", account_data["account_owner"])

    st.markdown("**Contacts**")
    contacts_df = pd.DataFrame(account_data["contacts"])[
        ["name", "role", "contact_type", "engagement_level", "notes"]
    ]
    st.dataframe(contacts_df, hide_index=True, width="stretch")

    with st.expander("Account notes", expanded=False):
        st.write(account_data["notes"])

if opp_data:
    st.divider()
    st.subheader("Opportunity")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stage", opp_data["stage"])
    col2.metric("Deal Value", f"₹{opp_data['deal_value_inr']:,}")
    col3.metric("Win Probability", f"{opp_data['win_probability_pct']}%")
    col4.metric("Expected Close", opp_data["expected_close_date"])

    band = opp_data.get("negotiation_band")
    if band:
        st.caption(
            f"Discount policy for this deal: {band['approved_min_discount_pct']}-"
            f"{band['approved_max_discount_pct']}% approved band, "
            f"{band['hard_ceiling_discount_pct']}% hard ceiling "
            f"(VP approval above {band['requires_vp_sales_approval_above_pct']}%, "
            f"CRO approval above {band['requires_cro_approval_above_pct']}%)"
        )

    st.markdown("**Line items**")
    if opp_data.get("line_items"):
        items_df = pd.DataFrame(opp_data["line_items"])[["sku", "description", "amount_inr"]]
        items_df["amount_inr"] = items_df["amount_inr"].apply(lambda v: f"₹{v:,}")
        st.dataframe(items_df, hide_index=True, width="stretch")

    st.markdown("**Risk factors**")
    risk_factors = opp_data.get("risk_factors", [])
    if risk_factors:
        for rf in risk_factors:
            icon = SEVERITY_ICON.get(rf["severity"], "⚪")
            st.markdown(f"{icon} **{rf['type'].replace('_', ' ').title()}** ({rf['severity']}) &mdash; {rf['description']}")
    else:
        st.success("No risk factors on file for this account -- healthy, low-risk deal.")

    with st.expander("Opportunity notes", expanded=False):
        st.write(opp_data.get("notes", ""))

# --- Documents: emails, transcripts, playbooks & catalog --------------------

st.divider()

email_hits = sorted(by_type.get("email", []), key=lambda h: h["date"] or "")
transcript_hits = sorted(by_type.get("transcript", []), key=lambda h: h["date"] or "")
playbook_hits = by_type.get("playbook", [])
catalog_hits = by_type.get("product_catalog", [])

tab_emails, tab_transcripts, tab_reference = st.tabs(
    [f"📧 Emails ({len(email_hits)})", f"📞 Call Transcripts ({len(transcript_hits)})", "📘 Company-Wide Reference"]
)


def _doc_title(hit: dict, fallback: str) -> str:
    for line in hit["text"].splitlines():
        line = line.strip()
        if line.startswith("**Subject:**"):
            return f"{hit['date']} -- {line.split('**Subject:**', 1)[1].strip()}"
        if line.startswith("**Type:**"):
            return f"{hit['date']} -- {line.split('**Type:**', 1)[1].strip()}"
    return fallback


with tab_emails:
    if not email_hits:
        st.info("No emails on file for this account.")
    else:
        labels = [_doc_title(h, h["date"] or "email") for h in email_hits]
        selected = st.radio("Select an email", range(len(email_hits)), format_func=lambda i: labels[i])
        st.markdown("---")
        st.markdown(email_hits[selected]["text"])

with tab_transcripts:
    if not transcript_hits:
        st.info("No call transcripts on file for this account.")
    else:
        labels = [_doc_title(h, h["date"] or "transcript") for h in transcript_hits]
        selected = st.radio("Select a transcript", range(len(transcript_hits)), format_func=lambda i: labels[i])
        st.markdown("---")
        st.markdown(transcript_hits[selected]["text"])

with tab_reference:
    st.caption("Shared across every account -- company-wide policy, playbooks, and the product catalog.")
    ref_docs: list[tuple[str, dict]] = []
    for h in playbook_hits:
        title = h["text"].splitlines()[0].lstrip("# ").strip() if h["text"].splitlines() else "Playbook"
        ref_docs.append((title, h))
    for h in catalog_hits:
        ref_docs.append(("Product Catalog", h))

    if not ref_docs:
        st.info("No reference documents found.")
    else:
        labels = [title for title, _ in ref_docs]
        selected = st.radio("Select a document", range(len(ref_docs)), format_func=lambda i: labels[i])
        title, hit = ref_docs[selected]
        st.markdown("---")
        if hit["doc_type"] == "product_catalog":
            catalog = json.loads(hit["text"])
            st.markdown(f"**{catalog['vendor_name']}** &mdash; catalog version {catalog['catalog_version']} ({catalog['currency']})")
            products_df = pd.DataFrame(catalog["products"])[
                ["sku", "name", "category", "pricing_model", "list_price", "unit"]
            ]
            products_df["list_price"] = products_df["list_price"].apply(lambda v: f"{catalog['currency']} {v:,}")
            st.dataframe(products_df, hide_index=True, width="stretch")
            with st.expander("Full product descriptions", expanded=False):
                for p in catalog["products"]:
                    st.markdown(f"**{p['name']}** (`{p['sku']}`)")
                    st.write(p["description"])
        else:
            st.markdown(hit["text"])
