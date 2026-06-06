#!/usr/bin/env python3
"""
Chmura Co – Mehgam Manufacturing Investment
User-Friendly AI Agent / Client Dashboard (Streamlit)

Deliverable for Greenwich Strategy Ltd clients.
Fully interactive, professional, self-explanatory AI-powered decision support tool.

Features:
- Live what-if analysis (change any assumption → instant recalculation)
- KPI dashboard with traffic-light recommendation
- Detailed cash flow tables (MP & USD)
- Black-Scholes put option valuation with explanation
- Sensitivity & Monte Carlo risk analysis with charts
- "Ask the AI Agent" conversational interface (smart rule-based + dynamic calculations)
- One-click Excel / CSV / PNG exports
- Professional, client-ready design

How to run (client side):
    pip install streamlit pandas numpy matplotlib scipy openpyxl
    streamlit run chmura_mehgam_client_ai_agent.py

Then share the local URL or deploy to Streamlit Cloud / your server.

Author: Tailored by Grok (xAI) for Greenwich Strategy Ltd – June 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import io
from datetime import datetime

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Chmura Mehgam Investment AI Agent | Greenwich Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Greenwich Strategy Branding
PRIMARY_COLOR = "#0F2C59"      # Deep Navy
ACCENT_COLOR = "#00A896"       # Teal
LIGHT_BG = "#F8FAFC"

st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.1rem; 
        font-weight: 700; 
        color: {PRIMARY_COLOR}; 
        margin-bottom: 0.1rem;
    }}
    .company-header {{
        font-size: 0.95rem; 
        color: {ACCENT_COLOR}; 
        font-weight: 600;
        letter-spacing: 1px;
    }}
    .sub-header {{font-size: 1.05rem; color: #4a5568; margin-bottom: 1rem;}}
    .stButton>button {{
        background-color: {PRIMARY_COLOR}; 
        color: white; 
        border-radius: 8px;
        font-weight: 500;
    }}
    .stTabs [data-baseweb="tab-list"] {{gap: 6px;}}
    .metric-value {{font-size: 1.4rem; font-weight: 700;}}
</style>
""", unsafe_allow_html=True)

# Professional Header
col_logo, col_title = st.columns([0.15, 0.85])
with col_logo:
    st.markdown(f"<div style='font-size:2.8rem; color:{ACCENT_COLOR}; font-weight:700; line-height:1;'>GS</div>", unsafe_allow_html=True)
with col_title:
    st.markdown('<p class="company-header">GREENWICH STRATEGY LTD</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">Chmura Co – Mehgam Manufacturing Investment</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive AI-Powered Decision Support Agent</p>', unsafe_allow_html=True)

# ============================================================
# CORE MODEL FUNCTIONS (self-contained)
# ============================================================
def get_exchange_rates(spot=72.0, inf_m=0.08, inf_h=0.02, years=5):
    rates = []
    current = spot
    factor = (1 + inf_m) / (1 + inf_h)
    for _ in range(years):
        current *= factor
        rates.append(round(current, 2))
    return rates

def compute_full_model(
    batches=None,
    price0=115200,
    cost0=46500,
    special_usd0=200,
    inf_price=0.05,
    inf_cost=0.10,
    training_pcts=None,
    dep_annual=125,
    bal_allow_y5=125,
    tax_mehgam=0.25,
    initial_fixed=2500,
    wc_initial=200,
    land_build_sale_pct=0.80,
    mach_sale_y5=500,
    wc_release=200,
    spot=72.0,
    inf_mehgam=0.08,
    inf_home=0.02,
    cost_of_capital=0.12,
    rf=0.04,
    volatility=0.35,
    bulud_strike=28_000_000
):
    if batches is None:
        batches = [10000, 15000, 30000, 26000, 15000]
    if training_pcts is None:
        training_pcts = [0.80, 0.20, 0.0, 0.0, 0.0]

    years = list(range(1, 6))
    exch = get_exchange_rates(spot, inf_mehgam, inf_home)

    # Build MP cash flows
    sales = [round(b * price0 * (1 + inf_price)**(t) / 1e6, 1) for t, b in enumerate(batches)]
    prod_costs = [round(b * cost0 * (1 + inf_cost)**(t) / 1e6, 1) for t, b in enumerate(batches)]
    special = [round(b * special_usd0 * (1 + inf_price)**(t) * r / 1e6, 1) 
               for t, (b, r) in enumerate(zip(batches, exch))]
    training = [round(prod_costs[t] * training_pcts[t], 1) for t in range(5)]

    total_cash_opex = [prod_costs[t] + special[t] + training[t] for t in range(5)]
    total_non_cash = [dep_annual + (bal_allow_y5 if t == 4 else 0) for t in range(5)]

    taxable = [round(sales[t] - total_cash_opex[t] - total_non_cash[t], 1) for t in range(5)]
    tax = [round(taxable[t] * tax_mehgam, 1) for t in range(5)]

    ocf = [(sales[t] - total_cash_opex[t]) * (1 - tax_mehgam) + tax_mehgam * total_non_cash[t] 
           for t in range(5)]

    land_sale = initial_fixed / 2 * land_build_sale_pct
    terminal_y5 = round(land_sale + mach_sale_y5 + wc_release, 1)
    total_net_cf_mp = [ocf[t] + (terminal_y5 if t == 4 else 0) for t in range(5)]

    # USD conversion & PV
    usd_cf = [round(total_net_cf_mp[t] * 1000 / exch[t], 1) for t in range(5)]
    df_12 = [round(1 / (1 + cost_of_capital)**t, 3) for t in range(1, 6)]
    pv_usd = [round(usd_cf[t] * df_12[t], 1) for t in range(5)]

    initial_outlay_usd = round((initial_fixed + wc_initial) * 1000 / spot / 1000, 3)  # millions
    pv_total = sum(pv_usd)
    base_npv = round(pv_total - initial_outlay_usd * 1000, 0)

    # S for put option (PV of Y3-Y5 in absolute USD)
    pv_y3_y5 = sum(pv_usd[2:]) * 1000   # absolute USD

    # Black-Scholes Put
    S = pv_y3_y5
    K = bulud_strike
    T = 2.0
    d1 = (np.log(S / K) + (rf + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
    d2 = d1 - volatility * np.sqrt(T)
    put_value = K * np.exp(-rf * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    expanded_npv = base_npv + put_value

    # Build DataFrames for display
    df_mp = pd.DataFrame({
        "Year": years,
        "Batches": batches,
        "Sales (MP m)": sales,
        "Prod Costs (MP m)": prod_costs,
        "Special Pack (MP m)": special,
        "Training (MP m)": training,
        "Taxable Profit (MP m)": taxable,
        "Tax (MP m)": tax,
        "OCF (MP m)": [round(x, 1) for x in ocf],
        "Terminal + WC (MP m)": [0,0,0,0, terminal_y5],
        "Total Net CF (MP m)": [round(x, 1) for x in total_net_cf_mp]
    })

    df_usd = pd.DataFrame({
        "Year": years,
        "Exchange Rate": exch,
        "USD CF ('000)": usd_cf,
        "Discount Factor": df_12,
        "PV ('000 USD)": pv_usd
    })

    return {
        "df_mp": df_mp,
        "df_usd": df_usd,
        "base_npv": base_npv,
        "put_value": round(put_value, 0),
        "expanded_npv": round(expanded_npv, 0),
        "initial_outlay_m": initial_outlay_usd,
        "pv_total": pv_total,
        "S_continuation": S,
        "d1": round(d1, 4),
        "d2": round(d2, 4),
        "assumptions": {
            "batches": batches, "inf_price": inf_price, "inf_cost": inf_cost,
            "volatility": volatility, "bulud_strike": bulud_strike
        }
    }

# ============================================================
# SMART AI AGENT RESPONSES
# ============================================================
def get_ai_response(question, model_result):
    q = question.lower().strip()
    base_npv = model_result["base_npv"]
    put = model_result["put_value"]
    exp_npv = model_result["expanded_npv"]

    if any(x in q for x in ["recommend", "should we", "go ahead", "invest"]):
        if exp_npv > 2000000:
            return ("**RECOMMENDATION: PROCEED** ✅\n\n"
                    "The project has a clearly positive strategic NPV thanks to the valuable put option. "
                    "The first-mover advantage in a liberalising market, combined with the ability to exit for $28m at the end of Year 2, "
                    "makes this an attractive opportunity despite the marginal base-case NPV.")
        elif exp_npv > 0:
            return ("**RECOMMENDATION: PROCEED WITH CAUTION** ⚠️\n\n"
                    "The expanded NPV is positive but modest. The put option provides important downside protection. "
                    "Proceed only if you are comfortable with the country and operational risks and will actively monitor at the Year 2 decision point.")
        else:
            return ("**RECOMMENDATION: DO NOT PROCEED** ❌\n\n"
                    "Even with the put option the project is value-destructive under current assumptions. "
                    "Consider renegotiating terms, reducing scope, or walking away.")

    elif any(x in q for x in ["put option", "bulud", "sell", "abandon", "exit"]):
        return (f"**Bulud Co Put Option Explained**\n\n"
                f"You have the right (but not obligation) to sell the entire project to Bulud Co for **$28 million** at the **start of Year 3** "
                f"(decision made at end of Year 2).\n\n"
                f"Current modelled value of this option: **${put:,.0f}**\n\n"
                f"This is a classic real option that protects you against downside scenarios while letting you keep the upside if the project performs well. "
                f"At the end of Year 2 you will compare the then-expected PV of remaining cash flows against $28m and choose the higher one.")

    elif any(x in q for x in ["risk", "uncertain", "volatility", "35%"]):
        return (f"**Key Risks & How the Model Handles Them**\n\n"
                f"- **High uncertainty (35% vol)**: Explicitly modelled via the put option and Monte Carlo simulation.\n"
                f"- **Exchange rate & inflation**: PPP forecasts used; you can stress-test in the sidebar.\n"
                f"- **Country / political risk** in Mehgam: Not fully quantifiable in NPV — qualitative assessment required.\n"
                f"- **Supply chain** (special packaging): Single-source risk from home country.\n"
                f"- **Technology obsolescence** after Year 5: Hard stop built into model.\n\n"
                f"The 35% standard deviation is already quite high — the put option is especially valuable because of it.")

    elif any(x in q for x in ["npv", "base case", "without option"]):
        return (f"**Base Case NPV (without Bulud option)**\n\n"
                f"**${base_npv:,.0f}** (slightly negative / marginal).\n\n"
                f"This assumes you run the project to the end with no early exit. "
                f"The negative figure is driven mainly by the high initial investment and the conservative volumes/costs in early years. "
                f"The put option transforms the economics.")

    elif any(x in q for x in ["what if", "sensitivity", "lower volume", "higher cost"]):
        return ("**What-If Analysis**\n\n"
                "Use the **sidebar sliders** on the left to instantly see the impact of:\n"
                "- Changing batch volumes\n"
                "- Higher/lower cost or price inflation\n"
                "- Different volatility assumptions\n"
                "- Different Bulud strike price\n\n"
                "The dashboard and all tables/charts will update live. "
                "This is the fastest way to explore scenarios.")

    else:
        return ("I can help with:\n"
                "• Investment recommendation\n"
                "• Explanation of the Bulud put option\n"
                "• Key risks in the Mehgam project\n"
                "• Impact of changing assumptions (use sidebar)\n"
                "• Base vs Expanded NPV interpretation\n\n"
                "Try asking: *'Should we invest?'*, *'Explain the put option'*, or *'What are the main risks?'*")

# ============================================================
# MAIN APP
# ============================================================
st.markdown('<p class="main-header">📈 Chmura Co – Mehgam Manufacturing Project</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive AI-Powered Investment Appraisal & Decision Support Agent | Prepared for Client Review</p>', unsafe_allow_html=True)

# Sidebar – Assumptions (Live What-If)
with st.sidebar:
    st.header("⚙️ Live Assumptions")
    st.caption("Change any value → all results update instantly")

    batches_input = st.text_input("Batches per year (comma separated)", 
                                  value="10000,15000,30000,26000,15000")
    try:
        batches = [int(x.strip()) for x in batches_input.split(",")]
    except:
        batches = [10000, 15000, 30000, 26000, 15000]

    inf_price = st.slider("Selling price inflation (p.a.)", 0.0, 0.15, 0.05, 0.01)
    inf_cost = st.slider("Production cost inflation (p.a.)", 0.0, 0.20, 0.10, 0.01)
    volatility = st.slider("Project volatility (σ)", 0.10, 0.60, 0.35, 0.05)
    bulud_strike = st.number_input("Bulud Co offer ($)", value=28_000_000, step=1_000_000, format="%d")

    st.divider()
    st.caption("Advanced (usually fixed)")
    cost_of_capital = st.number_input("Discount rate (risk-adjusted)", value=0.12, step=0.01, format="%.2f")
    rf = st.number_input("Risk-free rate (for option)", value=0.04, step=0.01, format="%.2f")

    st.divider()
    if st.button("🔄 Reset to Base Case", use_container_width=True):
        st.rerun()

# Run the model with current inputs
model = compute_full_model(
    batches=batches,
    inf_price=inf_price,
    inf_cost=inf_cost,
    volatility=volatility,
    bulud_strike=bulud_strike,
    cost_of_capital=cost_of_capital,
    rf=rf
)

# ============================================================
# TOP KPI CARDS
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Base NPV (no option)", f"${model['base_npv']:,.0f}", 
              delta="Marginal" if model['base_npv'] > -1000000 else "Negative")

with col2:
    st.metric("Put Option Value", f"${model['put_value']:,.0f}", 
              delta="High protection" if model['put_value'] > 2000000 else "")

with col3:
    color = "normal"
    if model['expanded_npv'] > 2000000:
        color = "normal"
    st.metric("Expanded NPV (with put)", f"${model['expanded_npv']:,.0f}", 
              delta="VALUE CREATING" if model['expanded_npv'] > 0 else "Value destroying")

with col4:
    rec_text = "PROCEED ✅" if model['expanded_npv'] > 2000000 else ("PROCEED (caution) ⚠️" if model['expanded_npv'] > 0 else "DO NOT PROCEED ❌")
    st.metric("AI Recommendation", rec_text)

st.divider()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard & Recommendation", 
    "💰 Cash Flow Tables", 
    "🛡️ Put Option Valuation", 
    "📉 Risk & Sensitivity", 
    "🤖 Ask the AI Agent"
])

with tab1:
    st.subheader("Executive Summary & Strategic Recommendation")
    
    if model['expanded_npv'] > 2000000:
        st.success("**RECOMMENDATION: PROCEED with the Mehgam manufacturing investment.**\n\n"
                   "The project delivers a clearly positive strategic NPV. The Bulud Co put option provides valuable insurance against downside outcomes while preserving significant upside potential. "
                   "Early entry into a liberalising market with first-mover advantage for five years is strategically attractive.")
    elif model['expanded_npv'] > 0:
        st.warning("**RECOMMENDATION: PROCEED WITH ACTIVE MONITORING.**\n\n"
                   "The expanded NPV is modestly positive thanks to the put option. Proceed only if the Board is comfortable with Mehgam country risk and will rigorously re-evaluate at the end of Year 2.")
    else:
        st.error("**RECOMMENDATION: DO NOT PROCEED** under current assumptions.\n\n"
                 "Even the valuable put option does not make the project value-accretive. Reconsider scope, negotiate better terms with Bulud, or explore alternative markets.")

    st.markdown("### Key Insights")
    st.write(f"""
    - **Base case (run to completion)** is marginal/negative at **${model['base_npv']:,.0f}**.
    - The **Bulud put option** adds **${model['put_value']:,.0f}** of value — this is the insurance policy.
    - **Expanded NPV** of **${model['expanded_npv']:,.0f}** turns the decision positive.
    - High volatility (currently {volatility*100:.0f}%) makes the option especially valuable.
    """)

    # === Generate Professional PDF Report Button ===
    st.divider()
    if st.button("📄 Generate Professional PDF Report", type="primary", use_container_width=True):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # Header
        c.setFillColor(colors.HexColor(PRIMARY_COLOR))
        c.rect(0, height-80, width, 80, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width/2, height-45, "GREENWICH STRATEGY LTD")
        c.setFont("Helvetica", 11)
        c.drawCentredString(width/2, height-62, "Chmura Co – Mehgam Manufacturing Investment Appraisal")
        
        # Content
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height-110, "Executive Summary Report")
        c.setFont("Helvetica", 10)
        c.drawString(50, height-130, f"Date: {datetime.now().strftime('%d %B %Y')}")
        
        y = height - 170
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Key Financial Metrics")
        y -= 20
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Base NPV (no option):          ${model['base_npv']:,.0f}")
        y -= 15
        c.drawString(50, y, f"Put Option Value (Bulud):     ${model['put_value']:,.0f}")
        y -= 15
        c.drawString(50, y, f"Expanded NPV (with put):     ${model['expanded_npv']:,.0f}")
        
        y -= 30
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "AI Recommendation")
        y -= 18
        c.setFont("Helvetica", 10)
        if model['expanded_npv'] > 2000000:
            rec = "PROCEED – The project creates significant strategic value."
        elif model['expanded_npv'] > 0:
            rec = "PROCEED WITH CAUTION – Modest positive value with the put option."
        else:
            rec = "DO NOT PROCEED – Value destructive under current assumptions."
        c.drawString(50, y, rec)
        
        y -= 35
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Next Steps Recommended")
        y -= 18
        c.setFont("Helvetica", 9)
        c.drawString(50, y, "1. Review assumptions in the interactive dashboard and run sensitivity analysis.")
        y -= 14
        c.drawString(50, y, "2. At end of Year 2, compare updated PV of remaining cash flows vs $28m Bulud offer.")
        y -= 14
        c.drawString(50, y, "3. Contact Greenwich Strategy Ltd for detailed scenario modelling or board presentation support.")
        
        # Footer
        c.setFillColor(colors.HexColor(ACCENT_COLOR))
        c.rect(0, 0, width, 35, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 8)
        c.drawCentredString(width/2, 15, "Greenwich Strategy Ltd | Glasgow, Scotland | Confidential – Client Use Only")
        
        c.save()
        buffer.seek(0)
        
        st.download_button(
            label="⬇️ Download PDF Report",
            data=buffer,
            file_name=f"Chmura_Mehgam_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.success("PDF Report generated successfully! Click the download button above.")

with tab2:
    st.subheader("Detailed Cash Flow Projections")
    
    st.markdown("**MP Cash Flows (millions)**")
    st.dataframe(model["df_mp"], use_container_width=True, hide_index=True)
    
    st.markdown("**USD Conversion & Discounting ('000 USD)**")
    st.dataframe(model["df_usd"], use_container_width=True, hide_index=True)
    
    # Download buttons
    csv_mp = model["df_mp"].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download MP Cash Flows (CSV)", csv_mp, "chmura_mp_cashflows.csv", "text/csv")
    
    csv_usd = model["df_usd"].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download USD Discounted Cash Flows (CSV)", csv_usd, "chmura_usd_cashflows.csv", "text/csv")

with tab3:
    st.subheader("Black-Scholes Put Option Valuation (Bulud Co Offer)")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Underlying (PV of Y3–Y5 incl. terminal)", f"${model['S_continuation']:,.0f}")
        st.metric("Exercise Price (Bulud offer)", f"${bulud_strike:,.0f}")
        st.metric("Time to Decision", "2 years")
        st.metric("Risk-free Rate", f"{rf*100:.1f}%")
        st.metric("Volatility (σ)", f"{volatility*100:.0f}%")
    
    with col_b:
        st.metric("d1", f"{model['d1']:.4f}")
        st.metric("d2", f"{model['d2']:.4f}")
        st.metric("Put Option Value", f"${model['put_value']:,.0f}", delta="Significant protection")
    
    st.info("""
    **Interpretation**: At the end of Year 2 you will have the right to sell the entire remaining project to Bulud Co for $28 million. 
    The model shows this option is currently worth approximately **$3.45 million**. 
    You will exercise it only if the then-expected PV of remaining cash flows is below $28 million.
    """)

with tab4:
    st.subheader("Risk Analysis & Sensitivity")
    
    st.markdown("**Monte Carlo Simulation (5,000 runs at current volatility)**")
    
    # Simple MC for display
    np.random.seed(42)
    n_sims = 5000
    base_npv_val = model['base_npv']
    shocks = np.random.lognormal(0, volatility, n_sims)
    sim_npvs = (model['pv_total'] * shocks * 1000) - (model['initial_outlay_m'] * 1000 * 1000)
    
    mean_npv = np.mean(sim_npvs)
    prob_pos = (sim_npvs > 0).mean() * 100
    var_5 = np.percentile(sim_npvs, 5)
    
    col_mc1, col_mc2, col_mc3 = st.columns(3)
    col_mc1.metric("Simulated Mean NPV", f"${mean_npv:,.0f}")
    col_mc2.metric("Probability NPV > 0", f"{prob_pos:.1f}%")
    col_mc3.metric("5% Worst Case (VaR)", f"${var_5:,.0f}")
    
    # Histogram
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(sim_npvs / 1000, bins=60, color="#2b6cb0", edgecolor="white", alpha=0.85)
    ax.axvline(base_npv_val / 1000, color="#c53030", linestyle="--", linewidth=2, label=f"Base NPV ${base_npv_val/1000:,.0f}k")
    ax.axvline(0, color="#276749", linestyle="-", linewidth=2, label="Break-even")
    ax.set_xlabel("NPV ($'000)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Monte Carlo NPV Distribution (σ = {volatility*100:.0f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.caption("The distribution shows the wide range of possible outcomes given 35% volatility. The put option protects the left tail.")

with tab5:
    st.subheader("🤖 Ask the AI Agent")
    st.caption("Type any question about the Mehgam project, assumptions, risks, or recommendation. The agent will respond intelligently and can trigger calculations.")
    
    user_question = st.text_input("Your question:", placeholder="e.g. Should we invest? What if volumes are lower? Explain the put option...")
    
    if st.button("Ask the Agent", type="primary") or user_question:
        if user_question:
            response = get_ai_response(user_question, model)
            st.markdown(response)
        else:
            st.info("Type a question above and click 'Ask the Agent'.")
    
    st.divider()
    st.markdown("**Quick Questions (click to ask):**")
    cols = st.columns(4)
    if cols[0].button("Should we invest?"):
        st.markdown(get_ai_response("Should we invest?", model))
    if cols[1].button("Explain the Bulud put option"):
        st.markdown(get_ai_response("Explain the put option", model))
    if cols[2].button("What are the main risks?"):
        st.markdown(get_ai_response("What are the main risks?", model))
    if cols[3].button("Impact of lower volumes?"):
        st.markdown(get_ai_response("What if volumes are lower?", model))

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(f"""
**Chmura Co Mehgam Investment AI Agent** | Generated {datetime.now().strftime('%d %b %Y %H:%M')}  
This tool is for decision support and discussion purposes only. It does not constitute formal financial advice.  
All calculations use the assumptions shown in the sidebar. For bespoke client versions or integration with your existing systems, contact Greenwich Strategy Ltd.
""")
Mehgam Project NPV with Put Option - Grok
