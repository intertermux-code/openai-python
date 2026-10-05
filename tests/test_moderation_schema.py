from __future__ import annotations

from openai.types.moderation import Categories, CategoryScores


def test_category_scores_accept_null_illicit_fields() -> None:
    """Regression test for https://github.com/openai/openai-python/issues/1786.

    The moderation API returns null for the `illicit` and `illicit_violent`
    scores instead of floats; the schema must accept that.
    """
    scores = CategoryScores.model_validate(
        {
            "harassment": 0.000255020015174523,
            "harassment/threatening": 1.3588138244813308e-05,
            "hate": 2.8068381652701646e-05,
            "hate/threatening": 1.0663524108167621e-06,
            "illicit": None,
            "illicit/violent": None,
            "self-harm": 9.841909195529297e-05,
            "self-harm/instructions": 7.693658517382573e-06,
            "self-harm/intent": 7.031533459667116e-05,
            "sexual": 0.013590452261269093,
            "sexual/minors": 0.0031673426274210215,
            "violence": 0.00022930897830519825,
            "violence/graphic": 4.927426198264584e-05,
        }
    )

    assert scores.illicit is None
    assert scores.illicit_violent is None


def test_categories_accept_null_illicit_fields() -> None:
    categories = Categories.model_validate(
        {
            "harassment": False,
            "harassment/threatening": False,
            "hate": False,
            "hate/threatening": False,
            "illicit": None,
            "illicit/violent": None,
            "self-harm": False,
            "self-harm/instructions": False,
            "self-harm/intent": False,
            "sexual": False,
            "sexual/minors": False,
            "violence": False,
            "violence/graphic": False,
        }
    )

    assert categories.illicit is None
    assert categories.illicit_violent is None
