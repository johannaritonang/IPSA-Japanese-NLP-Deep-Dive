"""Helpers for the IPSA social-listening NLP notebook.

Kept in a module so the notebook reads as analysis, not plumbing:
loading and cleaning the export, Japanese tokenisation, theme tagging,
weighted log-odds, topic coherence, chart styling and English glosses.
"""
import os
import re
import tempfile
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from janome.tokenizer import Tokenizer

from themes import DICT

# ---------------------------------------------------------------- palette (matches the Power BI report)
PLUM, ROSE, PEACH, LILAC, GOLD, BERRY, MAUVE, BLUSH, INK, MUTED = (
    "#4A1D3A", "#E0607E", "#F7A072", "#A78BDB", "#E8B04B", "#9C2A5B", "#CDBAC6", "#FFF7F5", "#2B2230", "#7A6B76")
SENT_COLORS = {"Positive": ROSE, "Neutral": MAUVE, "Negative": PLUM}
QUESTION_COLORS = {"1 Triggers": ROSE, "2 Barriers": PLUM, "3 Brand image": LILAC, "4 Efficacy": PEACH, "5 Channels": GOLD}


def style_charts():
    """Matplotlib defaults + a Japanese-capable font (first one installed wins)."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "IPAexGothic", "Noto Sans JP"]:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        print("No Japanese font found: Japanese labels may show as boxes. Install e.g. Noto Sans CJK JP.")
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#BFB3BA",
                         "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white", "axes.facecolor": "white"})


# ---------------------------------------------------------------- loading
PLATFORM = {"twitter": "Twitter", "rating_review": "@cosme reviews", "lexisnexis": "News (LexisNexis)", "news": "News (online)",
            "facebook": "Facebook", "fbcomment": "Facebook", "instagram": "Instagram", "forum": "Forums", "blog": "Blogs"}
PROMO = ["応募", "キャンペーン", "#pr", "amazon.co.jp", "m.qoo10.jp", "rakuten.co.jp", "item.rakuten", "ランキング"]


def voice_of(platform_type, source_type, text_lower):
    """Who is talking: news media, retailers/promotions, or consumers."""
    if platform_type in ("lexisnexis", "news"):
        return "Media"
    if source_type in ("Paid Media", "Competitor Owned") or (any(k in text_lower for k in PROMO) and platform_type != "rating_review"):
        return "Retail & promo"
    return "Consumer"


def load_posts(path):
    d = pd.read_excel(path)
    d["text"] = d["Post Data"].fillna("").astype(str)
    d["date"] = pd.to_datetime(d["Published Date"].str.replace(" SGT", "", regex=False))
    d["platform"] = d["Platform Type"].map(PLATFORM).fillna("Other")
    d["sentiment"] = d["Sentiments"].str.title()
    d["voice"] = [voice_of(p, s, t.lower()) for p, s, t in zip(d["Platform Type"], d["Source Type"], d["text"])]
    d["engagement"] = d[["Like Count", "Share Count", "Comment Count"]].fillna(0).sum(axis=1)
    d["is_dup_text"] = d.duplicated("text", keep="first")
    return d


# ---------------------------------------------------------------- tokenisation
# skincare words the default IPADIC dictionary splits badly (保湿 -> 保 + 湿る, 化粧水 -> 化粧 + 水)
USER_WORDS = ["保湿", "保湿力", "化粧水", "薬用化粧水", "美容液", "乳液", "浸透力", "肌荒れ", "毛穴", "インナードライ", "デパコス", "ベスコス", "プチプラ",
              "敏感肌", "乾燥肌", "混合肌", "脂性肌", "テクスチャー", "シャバシャバ", "パシャパシャ", "ベタつき", "リピ買い", "ニキビ跡", "マスク荒れ",
              "肌診断", "ハリ", "ツヤ", "キメ", "化粧ノリ", "化粧のり", "コットンパック", "導入液", "ミニボトル", "アットコスメ", "ベストコスメ"]
# spelling variants folded together
SYNONYMS = {"リピ": "リピート", "リピ買い": "リピート", "オススメ": "おすすめ", "テクスチャ": "テクスチャー", "うるおう": "潤う", "潤い": "潤う",
            "さらう": "さらっと", "サラッ": "さらっと", "サラサラ": "さらさら", "化粧のり": "化粧ノリ", "アットコスメ": "@cosme"}
# the product's own name and function words carry no meaning for topics
BRAND_TERMS = ["イプサ", "ipsa", "ザ・タイムr", "ザタイムr", "タイムr", "アクア", "the time", "reset", "aqua", "タイムリセット"]
STOPWORDS = set("""する いる なる ある れる られる せる こと もの よう これ それ あれ ここ そこ さん ちゃん くん 的 感じ 思う 言う 見る くる 来る いう 方 時 今 日 中 前 後
    ない 無い 使える 感じる 出来る できる つける 入る わかる 分かる 多い 使い 試す 合う の ん さ 気 化粧 化粧水 水 スキンケア 肌 使う 使用 商品 購入 買う 目 私 自分
    一 二 三 年 月 本 回 個 円 ml 人 みる くれる しまう いく おく あげる もらう 他 為 等 頃 毎日 今回 何 全部 たくさん とても 少し 本当 すごい 良い いい よい""".split())

_udic = os.path.join(tempfile.gettempdir(), "ipsa_userdic.csv")
with open(_udic, "w", encoding="utf8") as f:
    f.write("\n".join(f"{w},名詞,{w}" for w in USER_WORDS))
_tokenizer = Tokenizer(_udic, udic_type="simpledic", udic_enc="utf8")


def normalise(text):
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = re.sub(r"https?://\S+|\S+\.(?:co\.jp|com|jp)/\S*|@\w+", " ", s)   # links and @handles
    return s.replace("&amp;", "&")


def morphemes(text):
    """Raw janome output (surface, base form, part of speech) for teaching examples."""
    return [(t.surface, t.base_form, t.part_of_speech.split(",")[0]) for t in _tokenizer.tokenize(normalise(text))]


def tokenize(text):
    """Content words in base form; runs of nouns are merged into compounds (乾燥 + 肌 -> 乾燥肌)."""
    out, nouns = [], []

    def flush():
        if nouns:
            out.append("".join(nouns))
            nouns.clear()

    for tk in _tokenizer.tokenize(normalise(text)):
        pos = tk.part_of_speech.split(",")
        base = tk.base_form if tk.base_form != "*" else tk.surface
        if pos[0] == "名詞" and pos[1] not in ("数", "非自立", "代名詞", "接尾", "副詞可能") and not re.fullmatch(r"[\W_\d]+", base):
            nouns.append(base)
            continue
        flush()
        if pos[0] in ("形容詞", "動詞") and pos[1] == "自立":
            out.append(base)
    flush()
    out = [SYNONYMS.get(w, w) for w in out]
    out = [w for w in out if w in ("me", "@cosme") or not re.fullmatch(r"[a-z0-9_.\-]+", w)]  # drop latin noise, keep the ME line
    return [w for w in out if len(w) > 1 and w not in STOPWORDS and not any(b in w for b in BRAND_TERMS)]


# ---------------------------------------------------------------- dictionary themes
def tag_themes(texts):
    """Boolean matrix: post x theme, using the same keyword dictionary as the Power BI model."""
    low = pd.Series(texts).str.lower()
    return pd.DataFrame({theme: low.apply(lambda s, kws=kws: any(k.lower() in s for k in kws)).values for _, theme, kws in DICT},
                        index=getattr(texts, "index", None))


THEME_QUESTION = {theme: q for q, theme, _ in DICT}


# ---------------------------------------------------------------- statistics
def umass_coherence(binary_doc_term, top_ids):
    """UMass topic coherence: higher (closer to 0) = top words really co-occur in posts."""
    B = binary_doc_term
    df = np.asarray(B.sum(axis=0)).ravel()
    score, pairs = 0.0, 0
    for i in range(1, len(top_ids)):
        for j in range(i):
            co = B[:, top_ids[i]].multiply(B[:, top_ids[j]]).sum()
            score += np.log((co + 1) / df[top_ids[j]])
            pairs += 1
    return score / pairs


def weighted_log_odds(docs_a, docs_b, docs_prior, min_count=8):
    """Monroe, Colaresi & Quinn (2008) log-odds with an informative Dirichlet prior.
    Returns a z-score per word: > 0 leans to group A, < 0 to group B; |z| > 1.96 ~ significant."""
    ca = Counter(w for t in docs_a for w in t)
    cb = Counter(w for t in docs_b for w in t)
    cp = Counter(w for t in docs_prior for w in t)
    na, nb, a0 = sum(ca.values()), sum(cb.values()), sum(cp.values())
    rows = []
    for w in set(ca) | set(cb):
        ya, yb, aw = ca[w], cb[w], cp[w]
        if ya + yb < min_count:
            continue
        delta = np.log((ya + aw) / (na + a0 - ya - aw)) - np.log((yb + aw) / (nb + a0 - yb - aw))
        rows.append((w, delta / np.sqrt(1 / (ya + aw) + 1 / (yb + aw)), ya, yb))
    return pd.DataFrame(rows, columns=["word", "z", "count_a", "count_b"]).sort_values("z").reset_index(drop=True)


# ---------------------------------------------------------------- English glosses for chart labels
GLOSS = {
    "乾燥": "dryness", "肌荒れ": "irritated skin", "ニキビ": "acne", "口コミ": "reviews", "クチコミ": "reviews", "浸透": "absorbs", "テクスチャー": "texture",
    "保湿": "moisturising", "潤う": "hydrated", "高い": "pricey/high", "リピート": "repurchase", "me": "ME line", "乳液": "emulsion", "保湿力": "moisture power",
    "効果": "effect", "使い切る": "use up", "乾燥肌": "dry skin", "マスク": "mask", "サンプル": "sample", "荒れる": "break out", "おすすめ": "recommend",
    "人気": "popular", "さらっと": "silky-light", "みずみずしい": "dewy", "敏感肌": "sensitive skin", "コスメ": "cosmetics", "値段": "price",
    "調子": "skin condition", "悩む": "struggle", "ベタつく": "sticky", "水分": "moisture", "評価": "rating", "刺激": "irritation", "プレゼント": "gift",
    "コスパ": "value", "シャバシャバ": "watery", "期待": "expectation", "コットン": "cotton pad", "早い": "fast", "好き": "love it", "とろみ": "thickness",
    "なくなる": "runs out", "ボトル": "bottle", "もつ": "lasts", "持つ": "lasts", "欲しい": "want", "聞く": "ask", "浸透力": "absorption", "季節": "season",
    "戻る": "come back", "嬉しい": "happy", "クリーム": "cream", "与える": "give", "素敵": "lovely", "気に入る": "like it", "不可": "bad (so-so)",
    "悪い": "bad", "楽しみ": "look forward", "試し": "trial", "作る": "create", "保水成分": "water-holding agent", "保護": "protect", "持続": "lasting",
    "整える": "balance", "楽天": "Rakuten", "サイト": "website", "情報": "info", "一覧": "list", "確認": "check", "経つ": "time passes", "オイル": "oil",
    "皮膚": "skin", "当たる": "win (giveaway)", "チェック": "check", "お気に入り": "favourite", "治る": "heal", "改善": "improve", "違う": "different",
    "効く": "works", "生理": "period", "悩ます": "trouble", "少ない": "little", "ケア": "care", "酷い": "severe", "負ける": "give in", "冬場": "winter",
    "レギュラー": "ME Regular", "行く": "go (counter)", "エクストラ": "ME Extra", "カウンセリング": "counselling", "肌診断": "skin diagnosis",
    "毛穴": "pores", "セット": "set", "普通": "ordinary", "起きる": "occur", "思い切る": "take the plunge", "変化": "change", "シンプル": "simple",
    "アットコスメベスコス": "@cosme Best Cosme", "ベスコス": "Best Cosme", "洗顔": "face wash", "美容": "beauty", "薬用化粧水": "medicated lotion",
    "化粧ノリ": "makeup grip", "美容液": "serum", "デパコス": "dept-store cosmetics", "@cosme": "@cosme", "amazon": "Amazon", "雑誌": "magazine",
    "付録": "magazine freebie", "美的": "Biteki magazine", "受賞": "award", "ベストコスメ": "Best Cosme", "効果的": "effective", "大好き": "love",
    "さらさら": "silky", "ハリ": "firmness", "キメ": "skin texture", "ツヤ": "glow", "パシャパシャ": "splashy", "肌トラブル": "skin trouble",
    "医薬部外": "quasi-drug", "購入品": "haul", "べたつく": "sticky", "うれしい": "happy", "使い心地": "feel", "高め": "a bit pricey", "アルビオン": "Albion (rival)", "拭き取る": "wipe-off", "教える": "advise", "ライン": "line (set)",
}


def g(word):
    """Word with its English gloss, for chart labels."""
    return f"{word}  ({GLOSS[word]})" if word in GLOSS else word


# ---------------------------------------------------------------- manual sentiment audit
# Hand labels for every vendor-"Negative" consumer post plus real complaints found among vendor neutral/positive posts.
# row = position in the raw export (0-based, header excluded). Also saved as data/manual sentiment audit.csv.
AUDIT_ROWS = [
    (5, 'Negative', 'Positive', 'Not sticky', 'vendor Negative'),
    (9, 'Negative', 'Positive', 'Fewer breakouts, will keep using', 'vendor Negative'),
    (29, 'Negative', 'Positive', 'No period breakouts while using it', 'vendor Negative'),
    (45, 'Negative', 'Negative', 'Felt no effect despite good reviews', 'vendor Negative'),
    (51, 'Negative', 'Neutral', 'Past usage mention', 'vendor Negative'),
    (67, 'Negative', 'Positive', 'Bought twice for acne; skin feels plump', 'vendor Negative'),
    (70, 'Negative', 'Neutral', 'Routine plan', 'vendor Negative'),
    (81, 'Negative', 'Positive', 'Switched to it for more moisture', 'vendor Negative'),
    (87, 'Negative', 'Positive', 'Absorbs fast, good moisture', 'vendor Negative'),
    (138, 'Negative', 'Neutral', 'Asks how it is', 'vendor Negative'),
    (174, 'Negative', 'Neutral', 'Haul post', 'vendor Negative'),
    (184, 'Negative', 'Positive', 'Hydration-only lotion that sinks in', 'vendor Negative'),
    (201, 'Negative', 'Negative', 'Nothing changed after a full bottle', 'vendor Negative'),
    (248, 'Negative', 'Positive', 'Came back to it; skin stays good', 'vendor Negative'),
    (252, 'Negative', 'Negative', 'Expected too much; meh', 'vendor Negative'),
    (257, 'Negative', 'Neutral', 'Product mention', 'vendor Negative'),
    (269, 'Negative', 'Positive', 'Repurchased countless times; heals breakouts', 'vendor Negative'),
    (279, 'Negative', 'Neutral', 'Wants it but fears price/breakouts', 'vendor Negative'),
    (324, 'Negative', 'Positive', 'No breakouts and moisturising', 'vendor Negative'),
    (329, 'Negative', 'Positive', 'Too good; great moisture', 'vendor Negative'),
    (342, 'Negative', 'Positive', 'Was really good', 'vendor Negative'),
    (363, 'Negative', 'Neutral', 'Asks how it is', 'vendor Negative'),
    (394, 'Negative', 'Positive', 'Simple but moisturising', 'vendor Negative'),
    (408, 'Negative', 'Neutral', 'Comment on bottle shape', 'vendor Negative'),
    (422, 'Negative', 'Neutral', 'Asks how long a bottle lasts', 'vendor Negative'),
    (444, 'Negative', 'Neutral', 'Asks for opinions', 'vendor Negative'),
    (463, 'Negative', 'Neutral', 'Bought a sample to test', 'vendor Negative'),
    (484, 'Negative', 'Neutral', 'Routine list', 'vendor Negative'),
    (485, 'Negative', 'Neutral', 'Routine list', 'vendor Negative'),
    (489, 'Negative', 'Positive', 'Will buy it next', 'vendor Negative'),
    (490, 'Negative', 'Negative', "Didn't suit; not moisturising", 'vendor Negative'),
    (515, 'Negative', 'Positive', 'Moist, not sticky, less redness', 'vendor Negative'),
    (532, 'Negative', 'Negative', 'Period acne came back after switching', 'vendor Negative'),
    (599, 'Negative', 'Positive', 'Light yet moisturising; bought full size', 'vendor Negative'),
    (623, 'Negative', 'Neutral', 'Routine list', 'vendor Negative'),
    (639, 'Negative', 'Positive', 'Bought for breakouts; low expectations met', 'vendor Negative'),
    (640, 'Negative', 'Neutral', 'Went to the counter for a routine reset', 'vendor Negative'),
    (646, 'Negative', 'Neutral', 'Haul post', 'vendor Negative'),
    (662, 'Negative', 'Neutral', 'Routine list', 'vendor Negative'),
    (695, 'Negative', 'Neutral', 'Using it up fast', 'vendor Negative'),
    (719, 'Negative', 'Positive', 'Pimples heal faster', 'vendor Negative'),
    (720, 'Negative', 'Positive', 'Breakouts stopped except during periods', 'vendor Negative'),
    (722, 'Negative', 'Neutral', 'Not used yet', 'vendor Negative'),
    (741, 'Negative', 'Neutral', 'Haul post', 'vendor Negative'),
    (747, 'Negative', 'Positive', 'Recommends it for dehydrated skin', 'vendor Negative'),
    (749, 'Negative', 'Neutral', 'Choosing between brands', 'vendor Negative'),
    (786, 'Negative', 'Positive', 'Uses it to avoid breakouts', 'vendor Negative'),
    (815, 'Negative', 'Positive', 'Lotion feels really good', 'vendor Negative'),
    (846, 'Negative', 'Neutral', 'Wonders if it is good', 'vendor Negative'),
    (934, 'Negative', 'Neutral', 'About a different IPSA product', 'vendor Negative'),
    (945, 'Negative', 'Negative', "Didn't stop pre-period acne", 'vendor Negative'),
    (978, 'Negative', 'Negative', 'No improvement; stopped using', 'vendor Negative'),
    (1016, 'Negative', 'Positive', 'Best combo; skin improved', 'vendor Negative'),
    (1087, 'Negative', 'Positive', 'Natural feel; will repurchase', 'vendor Negative'),
    (1094, 'Negative', 'Neutral', 'Wants it; price makes her hesitate', 'vendor Negative'),
    (1123, 'Negative', 'Negative', 'Felt sticky; no effect', 'vendor Negative'),
    (1201, 'Negative', 'Neutral', 'Neither good nor bad', 'vendor Negative'),
    (1203, 'Negative', 'Positive', "Breakouts gone; can't live without it", 'vendor Negative'),
    (1227, 'Negative', 'Positive', 'Bought magazine for the free sample', 'vendor Negative'),
    (1233, 'Negative', 'Neutral', 'Night routine list', 'vendor Negative'),
    (1261, 'Negative', 'Neutral', 'Selling unused samples', 'vendor Negative'),
    (1266, 'Negative', 'Neutral', 'Haul post', 'vendor Negative'),
    (80, 'Neutral', 'Negative', 'Not moisturising at all; wants to give it away', 'keyword search in non-negative posts'),
    (145, 'Positive', 'Negative', 'Just ordinary; stings a little', 'keyword search in non-negative posts'),
    (311, 'Positive', 'Negative', '2/5 stars: filmy feel later, poor value', 'keyword search in non-negative posts'),
    (470, 'Neutral', 'Negative', "Drying, not moisturising; won't repurchase", 'keyword search in non-negative posts'),
    (500, 'Neutral', 'Negative', 'No benefit, felt overpriced', 'keyword search in non-negative posts'),
    (767, 'Neutral', 'Negative', 'Little effect; not enough for pores or moisture', 'keyword search in non-negative posts'),
    (874, 'Positive', 'Negative', 'Not moisturising enough; rejected', 'keyword search in non-negative posts'),
    (1167, 'Neutral', 'Negative', "Didn't suit; causes acne", 'keyword search in non-negative posts'),
    (1189, 'Neutral', 'Negative', "Didn't suit; daily breakouts", 'keyword search in non-negative posts'),
]


def load_audit():
    return pd.DataFrame(AUDIT_ROWS, columns=["row", "vendor_label", "audited_label", "note", "sample"])


# English translations (by hand) of the audited posts shown in the notebook.
# Complaint type is only filled for the posts that really criticise the product.
TRANSLATIONS = {
    5: ("", "Not sticky!"),
    9: ("", "IPSA The Time R Aqua: my skin breaks out less easily now. I'll keep using it next year."),
    29: ("", "While using this I had no period breakouts (acne, redness) at all. My previous lotion sometimes stung during my period, but that didn't happen…"),
    67: ("", "I repurchased it twice when my acne was bad. The texture is like water so I wondered how it would work… but it sinks right in and my skin turns plump."),
    45: ("No visible effect", "The reviews are great, but I couldn't feel it on my skin, which is disappointing. My skin is dry and gets rough, so I bought it with high hopes… Do I have to pair it with ME from the same line? Half is still left…"),
    201: ("No visible effect", "I tried it because the reviews were good. I used up a whole bottle; it wasn't bad, but nothing in particular changed, so I can't really say. Smaller pores, no flaking, fewer wrinkles…"),
    252: ("No visible effect", "They said it refills moisture like a tank… I expected too much (lol). Even the sample was neither good nor bad, but I gave in to its reputation and bought it…"),
    978: ("No visible effect", "Used two bottles. My rough skin didn't improve. There was no irritation so I bought a second one, but I saw no benefit in continuing and stopped."),
    1123: ("No visible effect", "You can splash it on, but it felt a bit sticky. I didn't feel much effect."),
    500: ("No visible effect", "Used it up but felt no particular benefit, so it felt expensive too. Like… water? Disappointing for something always in the rankings."),
    767: ("No visible effect", "Bought the IPSA emulsion and The Time R Aqua as a set in early September. Both are fresh and light! For me, I didn't feel much effect. Not enough for pores or moisture…"),
    532: ("Breakouts / irritation", "After switching to HABA squalane + Orbis U lotion even my pre-period acne stopped, but the dry pores on my cheeks stayed. I switched to IPSA The Time R Aqua and the pre-period acne came back. Still a skincare nomad."),
    945: ("Breakouts / irritation", "Bought it because I kept reading that it works on acne. Result: maybe bad timing before my period, but I broke out as usual. It isn't an acne product, so I guess it can't be helped…"),
    1167: ("Breakouts / irritation", "Unfortunately it didn't suit my skin. It gives me acne."),
    1189: ("Breakouts / irritation", "It has several ingredients for rough skin and pores, and many say they stopped breaking out, so I bought it with high hopes. Conclusion: it didn't suit me… small bumps every day. So disappointing."),
    145: ("Breakouts / irritation", "This very famous lotion. Cheaper than other department-store brands, so easy to try. My impression: just ordinary. It seems to sting a little…"),
    490: ("Not moisturising enough", "I bought it before because the reviews were good, but it didn't suit me… It didn't feel moisturising… I used it a little and then stopped."),
    80: ("Not moisturising enough", "Maybe it's my skin type, but it doesn't moisturise at all, my skin is parched the next morning and I feel no effect. I've used it twice and honestly want to give it away."),
    470: ("Not moisturising enough", "A beauty advisor picked it for me. But it was drying and not moisturising, so I didn't repurchase. Price-wise I suppose it's affordable."),
    874: ("Not moisturising enough", "My usual Orbis Clear lotion started to sting, so I tried an IPSA sample I had at home. My face feels light, but my skin feels tight and not moisturised enough for me… Rejected."),
    311: ("Poor value", "★★☆☆☆ When my skin was badly irritated and dry as a desert, it was the only thing I could use. Once my skin was normal again it left a strange film and didn't suit me. Pricey; a half-size bottle would be just right."),
}


def translation(row):
    return TRANSLATIONS.get(row, ("", ""))[1]


def complaint_type(row):
    return TRANSLATIONS.get(row, ("", ""))[0]


def match_topics(H_ref, H_other):
    """Pair topics of two NMF runs by cosine similarity (Hungarian algorithm). Returns (order, similarities)."""
    from scipy.optimize import linear_sum_assignment
    A = H_ref / np.linalg.norm(H_ref, axis=1, keepdims=True)
    Bm = H_other / np.linalg.norm(H_other, axis=1, keepdims=True)
    sim = A @ Bm.T
    rows, cols = linear_sum_assignment(-sim)
    return cols, sim[rows, cols]
