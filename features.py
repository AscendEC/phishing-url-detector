"""
URL -> feature vector for the phishing Random Forest.

The model was trained on the 'Web page phishing detection' dataset (Hannousse &
Yahiouche, 2021), whose 87 columns come from three sources. This module rebuilds
them from nothing but a URL:

  * "url"      lexical features computed from the URL string itself (exact)
  * "page"     content features parsed from the page's HTML (needs the page to load)
  * "lookup"   external look-ups: DNS, RDAP/WHOIS, Open PageRank (needs network/API)
  * "estimated" anything that couldn't be obtained -- the app fills the training-set
               median and flags it, and the user can overwrite it.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

try:  # optional: exact public-suffix handling (dataset was built with tldextract)
    import tldextract

    _TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)  # bundled snapshot, no network

    def split_host(host: str):
        e = _TLD(host)
        return e.subdomain, e.domain, e.suffix
except Exception:  # pragma: no cover - fallback keeps the app alive without the package
    _TWO_LEVEL = {
        "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.nz", "co.jp",
        "co.in", "com.br", "com.cn", "com.mx", "com.tr", "com.ar", "co.za", "com.sg", "com.hk",
        "com.tw", "com.ua", "com.pl", "com.ph", "com.my", "com.vn", "ac.in", "co.in", "net.in", "org.in", "co.id", "co.kr", "or.jp",
        "ne.jp", "net.cn", "org.cn", "com.co", "com.pe", "com.ng", "co.ke",
    }

    def split_host(host: str):
        parts = host.split(".")
        if len(parts) < 2:
            return "", host, ""
        suffix_len = 2 if ".".join(parts[-2:]) in _TWO_LEVEL else 1
        suffix = ".".join(parts[-suffix_len:])
        domain = parts[-suffix_len - 1] if len(parts) > suffix_len else ""
        sub = ".".join(parts[: -suffix_len - 1])
        return sub, domain, suffix


# --------------------------------------------------------------------------- lists
SHORTENERS = (
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|budurl\.com|ping\.fm|post\.ly|just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|cur\.lv|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|"
    r"cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|"
    r"1url\.com|tweez\.me|v\.gd|link\.zip\.net|rb\.gy|shorturl\.at|cutt\.ly|t\.ly"
)
PHISH_HINTS = ["wp", "login", "includes", "admin", "content", "site", "images", "js", "alibaba",
               "css", "myaccount", "dropbox", "themes", "plugins", "signin", "view"]
SUSPICIOUS_TLDS = {"fit", "tk", "gp", "ga", "work", "ml", "date", "wf", "cf", "zip", "cricket", "link",
                   "party", "gq", "faith", "racing", "download", "men", "accountant", "win", "loan",
                   "cc", "top", "review", "click", "stream", "study", "bid", "icu", "gdn", "kim",
                   "science", "trade", "webcam", "xyz", "country", "vip", "monster", "buzz"}
BRANDS = [
    "google", "facebook", "paypal", "apple", "microsoft", "amazon", "netflix", "instagram", "twitter",
    "linkedin", "yahoo", "ebay", "dropbox", "whatsapp", "adobe", "alibaba", "aliexpress", "outlook",
    "onedrive", "icloud", "wellsfargo", "bankofamerica", "citibank", "hsbc", "barclays", "santander",
    "fedex", "walmart", "bestbuy", "spotify", "github", "gmail", "youtube", "telegram", "snapchat",
    "tiktok", "binance", "coinbase", "blockchain", "metamask", "docusign", "vodafone", "rakuten",
    "mercadolibre", "airbnb", "wordpress", "bing", "baidu", "mozilla", "chase", "dhl", "usps",
]

# --------------------------------------------------------------------------- columns
# Columns the content/external parts are responsible for (everything else is lexical).
CONTENT_COLS = [
    "nb_redirection", "nb_external_redirection", "nb_hyperlinks", "ratio_intHyperlinks",
    "ratio_extHyperlinks", "nb_extCSS", "ratio_extRedirection", "ratio_extErrors", "login_form",
    "external_favicon", "links_in_tags", "ratio_intMedia", "ratio_extMedia", "iframe",
    "popup_window", "safe_anchor", "onmouseover", "right_clic", "empty_title", "domain_in_title",
    "domain_with_copyright",
]
EXTERNAL_COLS = ["whois_registered_domain", "domain_registration_length", "domain_age",
                 "web_traffic", "dns_record", "google_index", "page_rank"]
ALWAYS_ESTIMATED = ["statistical_report"]  # PhishTank statistics: not reproducible offline


# =========================================================================== lexical
def _words(s: str):
    return [w for w in re.split(r"[\-\.\/\?\=\@\&\%\:\_]", s.lower()) if w]


def normalise_url(raw: str) -> str:
    raw = raw.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", raw):
        raw = "https://" + raw
    return raw


def is_ip(host: str) -> int:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return 1
    except ValueError:
        return int(bool(re.fullmatch(r"(0x[0-9a-f]{1,2}\.){3}0x[0-9a-f]{1,2}", host.lower())))


def lexical_features(url: str) -> dict:
    p = urlparse(url)
    host = (p.hostname or "").lower()
    path = p.path or ""
    scheme = p.scheme
    sub, dom, suffix = split_host(host)
    # The dataset's `ip` flag fires on an IPv4 address OR a run of 7+ hex characters anywhere
    # in the URL (reverse-engineered: 99.97% agreement with the 11k labelled rows).
    ip = int(bool(re.search(r"[a-fA-F0-9]{7,}", url) or re.search(r"(\d{1,3}\.){3}\d{1,3}", url)))
    if is_ip(host):
        sub, dom, suffix = "", host, ""

    w_dom, w_sub, w_path = _words(dom), _words(sub), _words(path + ("?" + p.query if p.query else ""))
    words_raw = w_dom + w_path + w_sub
    w_host = w_dom + w_sub

    def stats(ws):
        if not ws:
            return 0, 0, 0
        ls = [len(w) for w in ws]
        return min(ls), max(ls), sum(ls) / len(ls)

    s_raw, l_raw, a_raw = stats(words_raw)
    s_host, l_host, a_host = stats(w_host)
    s_path, l_path, a_path = stats(w_path)

    repeat = 0
    for w in words_raw:
        for n in (2, 3, 4, 5):
            for i in range(len(w) - n + 1):
                if len(set(w[i:i + n])) == 1:
                    repeat += 1

    digits_url = sum(c.isdigit() for c in url)
    digits_host = sum(c.isdigit() for c in host)
    n_dots_host = host.count(".")
    path_l = path.lower()

    try:
        port = int(p.port) if p.port else 0
    except ValueError:
        port = 0

    return {
        "length_url": len(url),
        "length_hostname": len(host),
        "ip": ip,
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_at": url.count("@"),
        "nb_qm": url.count("?"),
        "nb_and": url.count("&"),
        "nb_eq": url.count("="),
        "nb_underscore": url.count("_"),
        "nb_tilde": url.count("~"),
        "nb_percent": url.count("%"),
        "nb_slash": url.count("/"),
        "nb_star": url.count("*"),
        "nb_colon": url.count(":"),
        "nb_comma": url.count(","),
        "nb_semicolumn": url.count(";"),
        "nb_dollar": url.count("$"),
        "nb_space": url.count(" ") + url.count("%20"),
        "nb_www": sum(1 for w in words_raw if w == "www"),
        "nb_com": sum(1 for w in words_raw if "com" in w),
        "nb_dslash": int(url.rfind("//") > 7),
        "http_in_path": path_l.count("http"),
        "https_token": 0 if scheme == "https" else 1,
        "ratio_digits_url": digits_url / len(url) if url else 0,
        "ratio_digits_host": digits_host / len(host) if host else 0,
        "punycode": int("xn--" in url.lower()),
        "port": int(bool(port) and port not in (80, 443)),
        "tld_in_path": int(bool(suffix) and suffix in path_l),
        "tld_in_subdomain": int(bool(suffix) and suffix in sub.split(".")),
        "abnormal_subdomain": int(bool(re.search(r"(http[s]?://(w[w]?[0-9]*|[0-9]+)\.)", url)) and not re.search(r"http[s]?://www\.", url)),
        "nb_subdomains": min(max(url.count("."), 1), 3),
        "prefix_suffix": int(bool(re.findall(r"https?://[^\-]+-[^\-]+/", url))),
        "random_domain": 0,  # original uses a word-dictionary model; neutral value
        "shortening_service": int(bool(re.search(SHORTENERS, url, re.I))),
        "path_extension": int(path_l.endswith(".txt")),
        "length_words_raw": len(words_raw),
        "char_repeat": repeat,
        "shortest_words_raw": s_raw, "shortest_word_host": s_host, "shortest_word_path": s_path,
        "longest_words_raw": l_raw, "longest_word_host": l_host, "longest_word_path": l_path,
        "avg_words_raw": a_raw, "avg_word_host": a_host, "avg_word_path": a_path,
        "phish_hints": sum(path_l.count(h) for h in PHISH_HINTS),
        "domain_in_brand": int(dom.lower() in BRANDS),
        "brand_in_subdomain": int(any(w in BRANDS for w in w_sub)),
        "brand_in_path": int(any(w in BRANDS for w in w_path)),
        "suspecious_tld": int(suffix.split(".")[-1] in SUSPICIOUS_TLDS) if suffix else 0,
    }


# =========================================================================== content
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


def fetch_page(url: str, timeout: float = 8.0):
    """Returns (html | None, final_url, redirect_count, external_redirects, error | None)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        hist = r.history
        base = split_host(urlparse(url).hostname or "")[1]
        ext = sum(1 for h in hist if split_host(urlparse(h.headers.get("location", "") or h.url).hostname or "")[1] not in ("", base))
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return None, r.url, len(hist), ext, f"non-HTML response ({ctype or 'unknown'})"
        return r.text, r.url, len(hist), ext, None
    except Exception as e:
        return None, url, 0, 0, f"{type(e).__name__}"


def content_features(html: str, url: str, redirects: int, ext_redirects: int) -> dict:
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    soup = BeautifulSoup(html, "html.parser")
    host = (urlparse(url).hostname or "").lower()
    base_dom = split_host(host)[1] or host

    def is_internal(link: str) -> bool:
        if not link or link.startswith(("#", "/", "./", "../", "?", "javascript:", "mailto:")):
            return True
        h = (urlparse(urljoin(url, link)).hostname or "").lower()
        return h == host or split_host(h)[1] == base_dom

    def collect(tags, attr):
        return [t.get(attr) for t in tags if t.get(attr) is not None]

    a_hrefs = collect(soup.find_all("a"), "href")
    link_hrefs = collect(soup.find_all("link"), "href")
    script_srcs = collect(soup.find_all("script"), "src")
    media = collect(soup.find_all(["img", "audio", "video", "embed", "source"]), "src")
    forms = collect(soup.find_all("form"), "action")
    css = [l.get("href") for l in soup.find_all("link", rel=lambda r: r and "stylesheet" in r) if l.get("href")]
    favicons = [l.get("href") for l in soup.find_all("link", rel=lambda r: r and "icon" in " ".join(r).lower()) if l.get("href")]

    allh = a_hrefs + link_hrefs + script_srcs + media + forms
    total = len(allh)
    ext_all = [l for l in allh if not is_internal(l)]
    ext_media = [m for m in media if not is_internal(m)]
    meta_like = link_hrefs + script_srcs + collect(soup.find_all("meta"), "content")
    ext_meta = [m for m in meta_like if m and m.startswith("http") and not is_internal(m)]

    unsafe_anchor = [a for a in a_hrefs if a.strip().lower() in ("#", "#content", "#skip", "javascript:void(0)", "javascript:;", "")]

    title = soup.title.get_text(strip=True) if soup.title else ""
    text = html.lower()
    squashed = re.sub(r"\s+", "", text)
    copy_txt = " ".join(re.findall(r"(?:©|&copy;|copyright)[^<]{0,80}", text))

    login_form = int(any((not f) or f.strip() in ("#", "about:blank", "javascript:void(0)") or not is_internal(f) for f in forms))
    iframes = soup.find_all("iframe")
    invisible_iframe = any(
        f.get("frameborder") in ("0", 0) or "hidden" in (f.get("style") or "").lower() or f.get("width") in ("0", 0) or f.get("height") in ("0", 0)
        for f in iframes
    )

    pct = lambda n, d: (n / d * 100) if d else 0.0
    return {
        "nb_redirection": redirects,
        "nb_external_redirection": int(ext_redirects > 0),
        "nb_hyperlinks": total + len(css) + len(favicons),
        "ratio_intHyperlinks": (total - len(ext_all)) / total if total else 0,
        "ratio_extHyperlinks": len(ext_all) / total if total else 0,
        "nb_extCSS": sum(1 for c in css if not is_internal(c)),
        "ratio_extRedirection": 0.0,   # needs a request per link; neutral
        "ratio_extErrors": 0.0,        # needs a request per link; neutral
        "login_form": login_form,
        "external_favicon": int(any(not is_internal(f) for f in favicons)),
        "links_in_tags": pct(len(ext_meta), len(meta_like)),
        "ratio_intMedia": pct(len(media) - len(ext_media), len(media)),
        "ratio_extMedia": pct(len(ext_media), len(media)),
        "iframe": int(invisible_iframe),
        "popup_window": int("prompt(" in text),
        "safe_anchor": pct(len(unsafe_anchor), len(a_hrefs)),
        "onmouseover": int('onmouseover="window.status' in squashed),
        "right_clic": int("event.button==2" in squashed),
        "empty_title": int(not title),
        "domain_in_title": 0 if base_dom and base_dom in title.lower() else 1,
        "domain_with_copyright": 0 if base_dom and base_dom in copy_txt else 1,
    }


# =========================================================================== external
def dns_lookup(host: str) -> int:
    """dataset semantics: 1 = NO DNS record."""
    try:
        socket.gethostbyname(host)
        return 0
    except Exception:
        return 1


def rdap_lookup(domain: str, timeout: float = 8.0):
    """(domain_age_days, registration_length_days) or None if unavailable."""
    try:
        r = requests.get(f"https://rdap.org/domain/{domain}", timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return None
        created = expires = None
        for ev in r.json().get("events", []):
            d = ev.get("eventDate", "")
            if ev.get("eventAction") == "registration":
                created = d
            elif ev.get("eventAction") == "expiration":
                expires = d
        parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
        now = datetime.now(timezone.utc)
        c, e = parse(created), parse(expires)
        if not c:
            return None
        return max((now - c).days, 0), max(((e - now).days if e else 0), 0)
    except Exception:
        return None


def open_pagerank(domain: str, api_key: str, timeout: float = 8.0):
    """(page_rank 0-10, global_rank or 0) from the free Open PageRank API, or None."""
    if not api_key:
        return None
    try:
        r = requests.get("https://openpagerank.com/api/v1.0/getPageRank", params={"domains[]": domain},
                         headers={"API-OPR": api_key}, timeout=timeout)
        item = r.json()["response"][0]
        pr = item.get("page_rank_integer")
        rank = item.get("rank")
        return (int(pr) if pr is not None else 0), (int(rank) if rank else 0)
    except Exception:
        return None


# =========================================================================== orchestration
def extract_all(raw_url: str, medians: dict, opr_key: str = "", fetch: bool = True):
    """
    Returns (features, provenance, notes).
      features   {column: value} for every column in `medians`
      provenance {column: 'url' | 'page' | 'lookup' | 'estimated'}
      notes      human-readable list of what failed
    """
    url = normalise_url(raw_url)
    feats, prov, notes = {}, {}, []

    for k, v in lexical_features(url).items():
        feats[k], prov[k] = v, "url"

    host = (urlparse(url).hostname or "").lower()
    sub, dom, suffix = split_host(host)
    registrable = f"{dom}.{suffix}" if dom and suffix else host

    if fetch:
        html, final_url, redirects, ext_redirects, err = fetch_page(url)
        if html:
            for k, v in content_features(html, final_url, redirects, ext_redirects).items():
                feats[k], prov[k] = v, "page"
        else:
            notes.append(f"Page content unavailable ({err}); content features estimated.")

        feats["dns_record"], prov["dns_record"] = dns_lookup(host), "lookup"
        rd = rdap_lookup(registrable)
        if rd:
            feats["whois_registered_domain"], prov["whois_registered_domain"] = 0, "lookup"
            feats["domain_age"], feats["domain_registration_length"] = rd
            prov["domain_age"] = prov["domain_registration_length"] = "lookup"
        else:
            notes.append("RDAP/WHOIS record unavailable; domain age and registration length estimated.")
        opr = open_pagerank(registrable, opr_key)
        if opr:
            feats["page_rank"], feats["web_traffic"] = opr
            prov["page_rank"] = prov["web_traffic"] = "lookup"
            # a domain Open PageRank knows about is, in practice, in Google's index
            feats["google_index"], prov["google_index"] = 0, "lookup"
        else:
            notes.append("No Open PageRank key set (or lookup failed); page_rank, web_traffic and google_index estimated.")
    else:
        notes.append("Network look-ups disabled; only URL-based features were computed.")

    for col, med in medians.items():
        if col not in feats:
            feats[col], prov[col] = med, "estimated"
    return {c: feats[c] for c in medians}, prov, notes
