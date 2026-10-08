import re
import os
import time
from playwright.sync_api import sync_playwright

CHINESE_CHAR_REGEX = re.compile(r"[\u4e00-\u9fff]")

def audit():
    os.makedirs("screenshots", exist_ok=True)
    findings = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        def check_elements(context_name):
            # Check document title
            title = page.title()
            if CHINESE_CHAR_REGEX.search(title):
                findings.append({
                    "context": context_name,
                    "target": "document.title",
                    "text": title,
                    "chinese": "".join(CHINESE_CHAR_REGEX.findall(title))
                })

            # Check text of all visible elements
            elements = page.query_selector_all("*")
            for el in elements:
                tag = el.evaluate("e => e.tagName.toLowerCase()")
                if tag in ["script", "style", "noscript", "svg", "path"]:
                    continue

                inner_text = el.evaluate("""e => {
                    const texts = [];
                    // Only direct child text or small containers
                    if (e.childNodes.length === 1 && e.childNodes[0].nodeType === 3) {
                        texts.push({type: 'textNode', val: e.textContent});
                    } else if (e.children.length === 0 && e.innerText) {
                        texts.push({type: 'innerText', val: e.innerText});
                    }
                    if (e.placeholder) texts.push({type: 'placeholder', val: e.placeholder});
                    if (e.title) texts.push({type: 'title', val: e.title});
                    return texts;
                }""")
                for item in inner_text:
                    val = item["val"]
                    if CHINESE_CHAR_REGEX.search(val):
                        el_desc = el.evaluate("e => e.outerHTML.substring(0, 100)")
                        findings.append({
                            "context": context_name,
                            "type": item["type"],
                            "element": f"<{tag}> {el_desc}",
                            "text": val[:120].strip(),
                            "chinese": "".join(set(CHINESE_CHAR_REGEX.findall(val)))
                        })

        print("1. Navigating to http://127.0.0.1:8765/?lang=en ...")
        page.goto("http://127.0.0.1:8765/?lang=en", wait_until="networkidle")
        time.sleep(1)

        # Tab 1: Identification Lab (initial)
        page.screenshot(path="screenshots/en_tab_lab_initial.png", full_page=True)
        check_elements("Tab 1 - Lab (Initial)")

        # Verify all probe dropdown options in English
        print("2. Verifying all probe dropdown options in English...")
        options = page.evaluate("""() => {
            const select = document.querySelector('.row-probe-select');
            return Array.from(select.options).map(o => ({val: o.value, text: o.text}));
        }""")
        for opt in options:
            if CHINESE_CHAR_REGEX.search(opt["text"]):
                findings.append({
                    "context": "Probe Options",
                    "target": f"option[{opt['val']}]",
                    "text": opt["text"],
                    "chinese": "".join(CHINESE_CHAR_REGEX.findall(opt["text"]))
                })

        # Add multiple probe rows with English outputs
        print("3. Adding English probe rows (Int, Color, RPS)...")
        # Row 1: arr_int5
        page.fill(".row-raw-text", "[17, 64, 92, 8, 41]")
        
        # Click Add Record for arr_color5
        page.click("button:has-text('Add Probe Record')")
        time.sleep(0.3)
        rows = page.query_selector_all(".submission-row")
        if len(rows) >= 2:
            select2 = rows[1].query_selector(".row-probe-select")
            select2.select_option("arr_color5")
            time.sleep(0.3)
            rows[1].query_selector(".row-raw-text").fill('["Red", "Blue", "Green", "Red", "Purple"]')

        # Click Add Record for arr_rps5
        page.click("button:has-text('Add Probe Record')")
        time.sleep(0.3)
        rows = page.query_selector_all(".submission-row")
        if len(rows) >= 3:
            select3 = rows[2].query_selector(".row-probe-select")
            select3.select_option("arr_rps5")
            time.sleep(0.3)
            rows[2].query_selector(".row-raw-text").fill('["Rock", "Scissors", "Paper", "Rock", "Scissors"]')

        page.screenshot(path="screenshots/en_tab_lab_inputs.png", full_page=True)

        # Run evaluation
        print("4. Executing statistical evaluation with multi-probe English responses...")
        page.click("#run-eval-btn")
        page.wait_for_selector("#results-area:not(.hidden)", timeout=8000)
        time.sleep(1)
        page.screenshot(path="screenshots/en_tab_lab_evaluated.png", full_page=True)
        check_elements("Tab 1 - Lab (Evaluated)")

        # Tab 2: Theory
        print("5. Checking Tab 2 - Theory...")
        page.click("#tab-theory-btn")
        time.sleep(1)
        page.screenshot(path="screenshots/en_tab_theory.png", full_page=True)
        check_elements("Tab 2 - Theory")

        # Tab 3: Database
        print("6. Checking Tab 3 - Database...")
        page.click("#tab-db-btn")
        time.sleep(1)
        page.screenshot(path="screenshots/en_tab_database.png", full_page=True)
        check_elements("Tab 3 - Database")

        # Open Import Modal
        print("7. Checking Import Modal...")
        page.click("#btn-import-archive")
        time.sleep(0.5)
        page.screenshot(path="screenshots/en_modal_import.png")
        check_elements("Modal - Import Archive")
        page.click("button:has-text('Cancel')")
        time.sleep(0.5)

        # Switch back to Tab 1 and test dynamic language switching (EN -> ZH -> EN)
        print("8. Testing dynamic toggle to ZH and back to EN...")
        page.click("#tab-lab-btn")
        time.sleep(0.5)
        # Click toggle to ZH
        page.click("#lang-toggle-btn")
        time.sleep(1)
        page.screenshot(path="screenshots/zh_tab_lab_evaluated.png", full_page=True)
        # Click toggle back to EN
        page.click("#lang-toggle-btn")
        time.sleep(1)
        page.screenshot(path="screenshots/en_tab_lab_after_toggle.png", full_page=True)
        check_elements("Tab 1 - Lab (After toggle back to EN)")

        browser.close()

    print("\n=================== AUDIT SUMMARY ===================")
    print(f"Total Chinese text occurrences flagged in English mode: {len(findings)}")
    # Deduplicate findings
    seen = set()
    deduped = []
    for f in findings:
        key = (f["context"], f.get("target", f.get("element", "")), f["text"])
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    if deduped:
        print(f"FAILED: Found {len(deduped)} Chinese snippets in English mode:")
        for i, f in enumerate(deduped, 1):
            print(f"\n[{i}] Context: {f['context']}")
            print(f"    Target: {f.get('target', f.get('element', ''))}")
            print(f"    Chinese characters: {f['chinese']}")
            print(f"    Text: {repr(f['text'])}")
    else:
        print("SUCCESS! Zero Chinese characters detected across all English views, controls, tooltips, and evaluations!")

if __name__ == "__main__":
    audit()
