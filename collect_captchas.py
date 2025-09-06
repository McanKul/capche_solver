import time, pathlib, sys, re
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

URL = "https://earsiv.migros.com.tr/giris.aspx"
SEL_IMG = "#imgCaptcha"
OUT_DIR = pathlib.Path("./dataset/raw")

COUNT = 1200          # kaç adet toplansın
HEADLESS = False       # debug için False yapabilirsin
PAUSE_MS = 120        # yenilemeler arası min bekleme

def wait_page_ready(page, timeout=15000):
    # sayfayı aç ve ağ sakinleşmesini bekle
    page.goto(URL, wait_until="domcontentloaded", timeout=timeout)
    try:
        page.wait_for_load_state("networkidle", timeout=8000)
    except PWTimeout:
        pass

    # olası KVKK/çerez butonları
    for label in ["Kabul", "Onayla", "Accept", "Tamam", "Anladım"]:
        try:
            page.get_by_role("button", name=re.compile(label, re.I)).click(timeout=1000)
            break
        except PWTimeout:
            continue

def wait_captcha_ready(page, timeout=15000):
    # img attach/visible
    img = page.locator(SEL_IMG)
    img.wait_for(state="attached", timeout=timeout)
    img.wait_for(state="visible", timeout=timeout)
    # gerçek yükleme (doğal boyut > 0)
    page.wait_for_function(
        """() => { const el = document.querySelector('#imgCaptcha');
                   return el && el.naturalWidth > 0 && el.naturalHeight > 0; }""",
        timeout=timeout
    )
    return img

def get_captcha_bytes(context, page):
    # src'yi al ve ham yanıtı indir
    src = page.locator(SEL_IMG).get_attribute("src")
    if not src:
        raise RuntimeError("imgCaptcha src boş döndü")
    full = urljoin(page.url, src)
    sep = "&" if urlparse(full).query else "?"
    full = f"{full}{sep}_ts={int(time.time()*1000)}"  # cache-buster
    resp = context.request.get(full)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} on {full}")
    return resp.body()

def refresh_captcha(page):
    # 1) Görünür "Resmi Yenile" linklerinden ilkini tıkla
    try:
        links = page.get_by_role("link", name="Resmi Yenile")
        n = links.count()
        for i in range(n):
            el = links.nth(i)
            if el.is_visible():
                el.click(timeout=1000)
                return True
    except Exception:
        pass

    # 2) JS fonksiyon varsa doğrudan çağır (sitede var)
    try:
        page.evaluate("() => window.RefreshImage && window.RefreshImage('imgCaptcha')")
        return True
    except Exception:
        pass

    # 3) Son çare: src'ye cache-buster ekle
    try:
        page.evaluate("""() => {
            const el = document.querySelector('#imgCaptcha');
            if (!el) return;
            const u = new URL(el.src, location.href);
            u.searchParams.set('_ts', Date.now().toString());
            el.src = u.toString();
        }""")
        return True
    except Exception:
        return False

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context()
        page = context.new_page()

        wait_page_ready(page)
        wait_captcha_ready(page)

        saved = 0
        tries  = 0
        while saved < COUNT:
            tries += 1
            try:
                # ham baytı indir ve kaydet
                data = get_captcha_bytes(context, page)
                (OUT_DIR / f"cap_{saved:05d}.png").write_bytes(data)
                saved += 1
                if saved % 50 == 0:
                    print(f"[INFO] Kaydedildi: {saved}/{COUNT}")
            except Exception as e:
                print(f"[WARN] indirme hatası: {e}")
                # sayfayı tazele
                try:
                    wait_page_ready(page)
                    wait_captcha_ready(page)
                except Exception as e2:
                    print(f"[WARN] sayfa yenile hatası: {e2}")
                    continue

            # yeni görsel
            ok = refresh_captcha(page)
            if not ok:
                # olmazsa sayfayı yenile
                wait_page_ready(page)
            else:
                # yeni captcha yüklenene kadar bekle
                try:
                    page.wait_for_function(
                        """() => { const el = document.querySelector('#imgCaptcha');
                                   return el && el.complete && el.naturalWidth > 0; }""",
                        timeout=8000
                    )
                except PWTimeout:
                    # yine de akışı kilitleme
                    pass
            page.wait_for_timeout(PAUSE_MS)

        browser.close()
    print(f"[DONE] Toplam {saved} captcha kaydedildi -> {OUT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FATAL:", e)
        sys.exit(1)
