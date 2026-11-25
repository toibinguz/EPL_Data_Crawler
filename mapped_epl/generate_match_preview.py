"""
mapped_epl/generate_match_preview.py
Generates a clean, self-contained 1:1 WhoScored interactive Match Centre page
from mapped_epl/whoscored_2627/2026-10-10_1983604_Manchester_United_vs_Tottenham.html.
"""

import gzip
import urllib.request
from pathlib import Path
from bs4 import BeautifulSoup, Comment

ROOT = Path(__file__).resolve().parent.parent
WS_HTML_PATH = ROOT / "mapped_epl" / "whoscored_2627" / "2026-10-10_1983604_Manchester_United_vs_Tottenham.html"
OUT_HTML = ROOT / "mapped_epl" / "match_preview_MUN_TOT_2026_10_10.html"

CORE_JS_KEYS = (
    "main-navigation-",
    "runtime-",
    "polyfills-",
    "vendor-",
    "whoscored-",
    "dust-full",
    "match-centre.js",
)


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip, identity"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
        if raw[:2] == b"\x1f\x8b" or resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", errors="replace")


def main():
    if OUT_HTML.exists():
        OUT_HTML.unlink()

    soup = BeautifulSoup(WS_HTML_PATH.read_text(encoding="utf-8", errors="replace"), "html.parser")

    # 1. Strip all HTML comments
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        c.extract()

    # 2. Remove ads, trackers, comments, footer; preserve any embedded require.config / matchCentreData scripts
    for sel in [
        ".left-sticky-absolute",
        ".right-sticky-absolute",
        ".sticky-unit-wrapper",
        "#user-comments",
        ".user-comments",
        "#footer-wrapper",
        "ins",
        "iframe",
        "noscript",
    ]:
        for el in soup.select(sel):
            for sc in el.find_all("script"):
                t = sc.string or ""
                if "require.config" in t or "matchCentreData" in t:
                    el.insert_before(sc.extract())
            el.decompose()

    # 3. Remove Live Stream tab & panel if non-functional
    for a in soup.select("#live-match-options a"):
        if "live stream" in a.get_text(strip=True).lower() or "livestream" in (a.get("href") or "").lower():
            if a.parent and a.parent.name == "li":
                a.parent.decompose()
    ls_panel = soup.find(id="livestream")
    if ls_panel:
        ls_panel.decompose()

    # 4. Inline WhoScored stylesheets directly into <style>
    for link in soup.find_all("link", rel=lambda v: v and "stylesheet" in v):
        href = link.get("href") or ""
        if "whoscored.com" in href or "cloudfront.net" in href:
            try:
                css = fetch_text(href)
                style_tag = soup.new_tag("style")
                style_tag.string = css
                link.replace_with(style_tag)
            except Exception as e:
                print(f"[!] CSS fetch skipped {href}: {e}")
                link.decompose()
        else:
            link.decompose()

    # 5. Keep core WhoScored JS bundles; remove ads/analytics/chatbot scripts
    for sc in soup.find_all("script"):
        src = sc.get("src") or ""
        if src:
            if not any(k in src for k in CORE_JS_KEYS):
                sc.decompose()
        else:
            txt = sc.string or ""
            if any(bad in txt for bad in ("googletagmanager", "gtag(", "googletag", "pluggedlabs", "oddschecker", "AiChatbot")):
                sc.decompose()

    # 6. Replace external team logos with clean inline SVG crests; remove external player/misc <img>
    mu_svg = '<svg viewBox="0 0 48 48" width="46" height="46" style="vertical-align:middle;"><circle cx="24" cy="24" r="22" fill="#da291c" stroke="#fbe122" stroke-width="2.5"/><path d="M16 14 L32 14 L30 34 L24 38 L18 34 Z" fill="#fbe122"/><text x="24" y="28" text-anchor="middle" fill="#da291c" font-size="11" font-weight="900">MU</text></svg>'
    tot_svg = '<svg viewBox="0 0 48 48" width="46" height="46" style="vertical-align:middle;"><circle cx="24" cy="24" r="22" fill="#ffffff" stroke="#132257" stroke-width="2.5"/><path d="M24 10 C28 16 29 22 25 30 L28 36 L20 36 L22 30 C19 23 20 16 24 10 Z" fill="#132257"/><circle cx="24" cy="36" r="3" fill="#132257"/></svg>'
    for img in soup.find_all("img"):
        src = (img.get("src") or "").lower()
        alt = (img.get("alt") or "").lower()
        if "32.png" in src or "manchester" in alt or "man utd" in alt:
            img.replace_with(BeautifulSoup(mu_svg, "html.parser"))
        elif "30.png" in src or "tottenham" in alt:
            img.replace_with(BeautifulSoup(tot_svg, "html.parser"))
        else:
            img.decompose()

    # 7. Hide non-working navigation/sign-in/breadcrumb/sub-nav bars cleanly (keep #header-wrapper in DOM for whoscored.js init)
    extra_style = soup.new_tag("style")
    extra_style.string = """
    #header-wrapper, #breadcrumb-nav, #sub-navigation, .teammate-widget { display: none !important; }
    body { background: #fff !important; margin: 0 !important; padding: 14px 0 32px 0 !important; }
    #layout-wrapper { margin: 0 auto !important; }
    """
    if soup.head:
        soup.head.append(extra_style)

    # 8. Live MutationObserver I18N fixer for local file:// rendering
    i18n_script = soup.new_tag("script")
    i18n_script.string = """
    window.liveStream = window.liveStream || { activate: function(){}, deactivate: function(){} };
    (function() {
      const I18N = {
        'stats_clmn_rating': 'Ratings', 'stats_clmn_shot_total': 'Total Shots',
        'stats_clmn_poss_perc': 'Possession%', 'stats_clmn_pass_succ': 'Pass Success%',
        'stats_clmn_dribble': 'Dribbles', 'stats_clmn_aerial_won': 'Aerials Won',
        'stats_clmn_tackle': 'Tackles', 'stats_clmn_corner': 'Corners',
        'stats_clmn_dispossessed': 'Dispossessed', 'stats_clmn_dispossesed': 'Dispossessed', 'stats_info_more': 'More',
        'live_info_last_update': 'Last updated: FT',
        'chalk_clmn_shot': 'Shots', 'chalk_clmn_pass': 'Passes', 'chalk_clmn_dribble': 'Dribbles',
        'chalk_clmn_tackle_total': 'Tackles Attempted', 'chalk_clmn_inter': 'Interceptions',
        'chalk_clmn_clear': 'Clearances', 'chalk_clmn_block': 'Blocks', 'chalk_clmn_offside': 'Offsides',
        'chalk_clmn_foul': 'Fouls', 'chalk_clmn_aerial': 'Aerial duels', 'chalk_clmn_touch': 'Touches',
        'chalk_clmn_poss_loss': 'Loss of possession', 'chalk_clmn_error': 'Errors',
        'chalk_clmn_save': 'Saves', 'chalk_clmn_claim': 'Claims', 'chalk_clmn_punch': 'Punches',
        'chalk_clmn_all': 'All', 'chalk_clmn_result': 'Result', 'chalk_clmn_zone': 'Zone',
        'chalk_clmn_situation': 'Situation', 'chalk_clmn_body_part': 'Body Part',
        'chalk_clmn_goal': 'Goals', 'chalk_clmn_shot_on': 'Shots on Target',
        'chalk_clmn_shot_off': 'Shots off Target', 'chalk_clmn_wood': 'Woodworks',
        'chalk_clmn_six_yard': 'Six Yard Box', 'chalk_clmn_pen_area': 'Penalty Area',
        'chalk_clmn_out_box': 'Out of Box', 'chalk_clmn_open_play': 'Open Play',
        'chalk_clmn_counter': 'Fast Break', 'chalk_clmn_set_piece': 'Set Piece',
        'chalk_clmn_penalty': 'Penalty', 'chalk_clmn_right_foot': 'Right Foot',
        'chalk_clmn_left_foot': 'Left Foot', 'chalk_clmn_head': 'Head', 'chalk_clmn_other': 'Other',
        'heatmap_clmn_touch:': 'Touches:', 'heatmap_clmn_touch': 'Touches'
      };
      function fixLabels(root) {
        const walker = document.createTreeWalker(root || document.body, NodeFilter.SHOW_TEXT, null, false);
        let n;
        while ((n = walker.nextNode())) {
          const t = n.nodeValue && n.nodeValue.trim();
          if (t && I18N[t]) n.nodeValue = n.nodeValue.replace(t, I18N[t]);
        }
      }
      document.addEventListener('DOMContentLoaded', () => {
        fixLabels();
        const obs = new MutationObserver(() => fixLabels());
        obs.observe(document.body, { childList: true, subtree: true });
      });
    })();
    """
    if soup.body:
        soup.body.append(i18n_script)

    OUT_HTML.write_text(str(soup), encoding="utf-8")
    print(f"[+] Built clean 1:1 WhoScored HTML: {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
