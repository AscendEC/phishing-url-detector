"""Best-effort extraction of the URL-based (lexical) features from a raw URL.

The model was trained on 81 features. Only the lexical ones can be computed
from the URL string alone. Content-based features (HTML) and external ones
(WHOIS, Google index, PageRank, traffic, DNS) need live lookups and must be
supplied by the user.

These are approximations of the original dataset's definitions, so values may
differ slightly from the training data.
"""
import re
from urllib.parse import urlparse

SHORTENERS = {
    "bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc",
}
PHISH_HINTS = ["wp", "login", "includes", "admin", "content", "site", "images",
               "js", "alibaba", "css", "myaccount", "dropbox", "themes",
               "plugins", "signin", "view"]
SUSPICIOUS_TLDS = {"zip", "country", "kim", "science", "work", "party", "gq",
                   "link", "click", "top", "xyz", "tk", "ml", "ga", "cf"}


def _words(s):
    return [w for w in re.split(r"[\W_]+", s) if w]


def _stats(words):
    if not words:
        return 0, 0, 0.0
    lens = [len(w) for w in words]
    return min(lens), max(lens), sum(lens) / len(lens)


def extract_url_features(url: str) -> dict:
    url = url.strip()
    parsed = urlparse(url if "//" in url else "http://" + url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    parts = host.split(".") if host else []
    tld = parts[-1] if parts else ""
    subdomains = parts[:-2] if len(parts) > 2 else []

    host_words, path_words, raw_words = _words(host), _words(path), _words(url)
    h_min, h_max, h_avg = _stats(host_words)
    p_min, p_max, p_avg = _stats(path_words)
    r_min, r_max, r_avg = _stats(raw_words)

    n_digits_url = sum(c.isdigit() for c in url)
    n_digits_host = sum(c.isdigit() for c in host)

    f = {
        "length_url": len(url),
        "length_hostname": len(host),
        "ip": int(bool(re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", host))),
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
        "nb_www": url.lower().count("www"),
        "nb_com": url.lower().count(".com"),
        "nb_dslash": url.count("//"),
        "http_in_path": int("http" in path.lower()),
        "https_token": int(parsed.scheme == "https"),
        "ratio_digits_url": n_digits_url / len(url) if url else 0,
        "ratio_digits_host": n_digits_host / len(host) if host else 0,
        "punycode": int("xn--" in host),
        "port": int(parsed.port is not None) if _safe_port(parsed) else 0,
        "tld_in_path": int(bool(tld) and tld in path.lower()),
        "tld_in_subdomain": int(bool(tld) and any(tld == s for s in subdomains)),
        "abnormal_subdomain": int(bool(re.match(r"^w{2,3}\d+\.", host))),
        "nb_subdomains": len(subdomains),
        "prefix_suffix": int("-" in host),
        "shortening_service": int(host in SHORTENERS),
        "path_extension": int(bool(re.search(r"\.(txt|exe|js)$", path.lower()))),
        "length_words_raw": len(raw_words),
        "char_repeat": _max_repeat(url),
        "shortest_words_raw": r_min,
        "shortest_word_host": h_min,
        "shortest_word_path": p_min,
        "longest_words_raw": r_max,
        "longest_word_host": h_max,
        "longest_word_path": p_max,
        "avg_words_raw": r_avg,
        "avg_word_host": h_avg,
        "avg_word_path": p_avg,
        "phish_hints": sum(url.lower().count(h) for h in PHISH_HINTS),
        "suspecious_tld": int(tld in SUSPICIOUS_TLDS),
    }
    return {k: float(v) for k, v in f.items()}


def _safe_port(parsed):
    try:
        parsed.port
        return True
    except ValueError:
        return False


def _max_repeat(s):
    best = run = 1 if s else 0
    for a, b in zip(s, s[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best
