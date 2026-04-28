custom_css = """
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

:root {
    --bg: #fafafa;
    --surface: #ffffff;
    --surface-hover: #f5f5f7;
    --border: #e5e5ea;
    --text-primary: #1d1d1f;
    --text-secondary: #6e6e73;
    --text-tertiary: #aeaeb2;
    --accent: #0071e3;
    --accent-hover: #0077ED;
    --green: #34c759;
    --gold: #f5a623;
    --silver: #8e8e93;
    --bronze: #af6e3d;
    --radius: 14px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 14px rgba(0,0,0,0.06), 0 2px 6px rgba(0,0,0,0.04);
}

/* ─── Global Reset ─── */
.gradio-container {
    max-width: 1120px !important;
    margin: 0 auto !important;
    background: var(--bg) !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    padding: 0 24px !important;
}

.gradio-container .main,
.gradio-container .contain {
    gap: 0 !important;
}

footer { display: none !important; }

/* ─── Header ─── */
#header-block {
    background: transparent !important;
    border: none !important;
    padding: 56px 0 32px !important;
    text-align: center;
}

#header-block h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 44px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 8px !important;
    line-height: 1.1 !important;
}

#header-block p {
    font-size: 17px !important;
    color: var(--text-secondary) !important;
    font-weight: 400 !important;
    margin: 0 !important;
    letter-spacing: -0.01em !important;
}

/* ─── Tabs ─── */
.tab-buttons {
    border: none !important;
}

.tab-buttons > div:first-child {
    background: transparent !important;
    border: none !important;
    justify-content: center !important;
    gap: 4px !important;
    padding: 0 0 28px !important;
}

.tab-buttons button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}

.tab-buttons button:hover {
    color: var(--text-primary) !important;
}

.tab-buttons button.selected {
    color: var(--text-primary) !important;
    border-bottom-color: var(--text-primary) !important;
    background: transparent !important;
}

/* ─── Leaderboard Table (HTML) ─── */
#leaderboard-table {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
    overflow: hidden !important;
    padding: 0 !important;
}

#leaderboard-table table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

#leaderboard-table thead th {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-tertiary);
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    white-space: nowrap;
    position: sticky;
    top: 0;
    background: var(--surface);
}

#leaderboard-table thead th:first-child {
    padding-left: 24px;
}

#leaderboard-table thead th.num {
    text-align: right;
}

#leaderboard-table tbody tr {
    transition: background 0.15s ease;
}

#leaderboard-table tbody tr:hover {
    background: var(--surface-hover);
}

#leaderboard-table tbody td {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: var(--text-primary);
    padding: 13px 16px;
    border-bottom: 1px solid #f0f0f2;
    white-space: nowrap;
}

#leaderboard-table tbody td:first-child {
    padding-left: 24px;
}

#leaderboard-table tbody td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-weight: 400;
    color: var(--text-secondary);
}

#leaderboard-table tbody td.avg-cell {
    font-weight: 600;
    color: var(--text-primary);
}

#leaderboard-table tbody tr:last-child td {
    border-bottom: none;
}

.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}

.rank-1 { background: linear-gradient(135deg, #ffd700, #f5a623); color: #7a5c00; }
.rank-2 { background: linear-gradient(135deg, #e8e8ed, #c7c7cc); color: #48484a; }
.rank-3 { background: linear-gradient(135deg, #deb887, #af6e3d); color: #5a3520; }
.rank-other { background: #f2f2f7; color: var(--text-secondary); }

.model-name {
    font-weight: 500;
    color: var(--text-primary);
}

.model-type-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 500;
    color: var(--text-tertiary);
    background: #f2f2f7;
    padding: 2px 8px;
    border-radius: 6px;
    margin-left: 8px;
    vertical-align: middle;
}

/* score color coding */
.score-high { color: #248a3d !important; font-weight: 500 !important; }
.score-mid { color: var(--text-primary) !important; }
.score-low { color: #ff3b30 !important; }

/* ─── Submit Tab ─── */
#submit-section {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
    padding: 40px !important;
}

#submit-section .gr-form,
#submit-section .form {
    border: none !important;
    background: transparent !important;
}

#submit-section input[type="text"],
#submit-section textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    background: var(--bg) !important;
    transition: border-color 0.2s ease !important;
}

#submit-section input[type="text"]:focus,
#submit-section textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.1) !important;
    outline: none !important;
}

#submit-section label span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

#submit-btn {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    background: var(--text-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

#submit-btn:hover {
    background: #333336 !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ─── Analytics Tab ─── */
#analytics-section {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
    padding: 32px !important;
}

/* ─── About / instructions ─── */
.instructions-block {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 32px 36px !important;
    margin-bottom: 24px !important;
}

.instructions-block h2 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-bottom: 16px !important;
}

.instructions-block p,
.instructions-block li {
    font-size: 15px !important;
    color: var(--text-secondary) !important;
    line-height: 1.65 !important;
}

.instructions-block code {
    background: #f2f2f7 !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    font-size: 13px !important;
}

/* ─── Misc Gradio overrides ─── */
.gr-group, .gr-box, .gr-panel {
    border: none !important;
    box-shadow: none !important;
}

.gr-padded {
    padding: 0 !important;
}

/* dropdown */
.gradio-dropdown {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* file upload */
.gr-file-upload, .upload-button {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--bg) !important;
}

/* plot override */
.js-plotly-plot {
    border-radius: 10px !important;
}

/* status output */
#submit-status textarea {
    font-family: 'DM Sans', monospace !important;
    background: #f2f2f7 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d1d1d6; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #aeaeb2; }
"""

get_window_url_params = """
    function(url_params) {
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return url_params;
    }
    """
