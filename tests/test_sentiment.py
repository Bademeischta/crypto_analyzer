import pytest

from src.features.sentiment import analyze_posts, classify_fear_greed, mentions, parse_atom_titles, score_title


@pytest.mark.parametrize(
    "title, expected",
    [
        ("SOL breaks out to new highs", True),
        ("Why $sol could double", True),
        ("Solana ecosystem update", True),  # Alias
        ("A new solution for scaling", False),  # kein Teilwort-Treffer
        ("console wars", False),
        ("I'm going solo on this one", False),
        ("sol is lowercase and ambiguous", False),
    ],
)
def test_mentions_uses_word_boundaries(title, expected):
    assert mentions(title, "SOL", ["Solana"]) is expected


def test_score_title_handles_punctuation():
    assert score_title("To the moon!!! 🚀") == 1
    assert score_title("Massive dump, total rug.") == -1
    assert score_title("Pump and dump?") == 0
    assert score_title("Weekly discussion") == 0


def test_analyze_posts_aggregates():
    posts = [
        {"title": "BTC to the moon", "score": 10},
        {"title": "BTC crash incoming", "score": 30},
        {"title": "BTC weekly thread", "score": 20},
        {"title": "ETH news", "score": 99},
    ]
    res = analyze_posts(posts, "BTC", [])
    assert res["post_count"] == 3
    assert res["bullish_score"] == pytest.approx(1 / 3, abs=1e-3)
    assert res["net_sentiment"] == 0.0
    assert res["avg_upvotes"] == 20.0
    assert res["top_titles"][0] == "BTC crash incoming"


def test_parse_atom_titles():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>First $BTC post</title></entry>
      <entry><title> Second post </title></entry>
    </feed>"""
    posts = parse_atom_titles(xml, "CryptoCurrency")
    assert [p["title"] for p in posts] == ["First $BTC post", "Second post"]
    assert posts[0]["source"] == "rss" and posts[0]["score"] is None


@pytest.mark.parametrize("value, label", [(10, "Extreme Fear"), (40, "Fear"), (50, "Neutral"), (70, "Greed"), (90, "Extreme Greed")])
def test_classify_fear_greed(value, label):
    assert classify_fear_greed(value) == label
