"""Basic content-based recommendation system using scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Recommendation:
	"""A recommended item and its similarity score."""

	title: str
	score: float


class ContentRecommender:
	"""Recommend items with similar text descriptions."""

	def __init__(self, items: pd.DataFrame) -> None:
		required_columns = {"title", "description"}
		missing_columns = required_columns.difference(items.columns)
		if missing_columns:
			missing = ", ".join(sorted(missing_columns))
			raise ValueError(f"Missing required columns: {missing}")
		if items.empty:
			raise ValueError("The catalog cannot be empty")

		self.items = items.reset_index(drop=True).copy()
		self.vectorizer = TfidfVectorizer(stop_words="english")
		descriptions = self.items["description"].fillna("")
		self.item_matrix = self.vectorizer.fit_transform(descriptions)
		self.similarity_matrix = cosine_similarity(self.item_matrix)

	def recommend(self, title: str, number_of_recommendations: int = 5) -> list[Recommendation]:
		"""Return the most similar items for a catalog title."""
		if number_of_recommendations < 1:
			raise ValueError("number_of_recommendations must be at least 1")

		matching_items = self.items.index[self.items["title"].str.casefold() == title.casefold()]
		if matching_items.empty:
			available_titles = ", ".join(self.items["title"])
			raise ValueError(f"Unknown title: {title}. Available titles: {available_titles}")

		item_index = matching_items[0]
		ranked_indices = self.similarity_matrix[item_index].argsort()[::-1]
		recommendations = []
		for index in ranked_indices:
			if index == item_index:
				continue
			recommendations.append(
				Recommendation(
					title=self.items.iloc[index]["title"],
					score=float(self.similarity_matrix[item_index, index]),
				)
			)
			if len(recommendations) == number_of_recommendations:
				break
		return recommendations


def build_demo_catalog() -> pd.DataFrame:
	"""Create a small catalog so the example runs without external data."""
	return pd.DataFrame(
		[
			{"title": "The Matrix", "description": "science fiction action virtual reality hacker"},
			{"title": "Inception", "description": "science fiction thriller dreams mind action"},
			{"title": "Interstellar", "description": "science fiction space exploration time drama"},
			{"title": "The Dark Knight", "description": "action crime superhero drama batman"},
			{"title": "Toy Story", "description": "animation comedy toys friendship family"},
			{"title": "Finding Nemo", "description": "animation adventure ocean family friendship"},
		]
	)


if __name__ == "__main__":
	recommender = ContentRecommender(build_demo_catalog())
	selected_title = "The Matrix"
	print(f"Recommendations for '{selected_title}':")
	for recommendation in recommender.recommend(selected_title, number_of_recommendations=3):
		print(f"- {recommendation.title} (similarity: {recommendation.score:.2f})")
