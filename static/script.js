document.addEventListener("DOMContentLoaded", () => {
	const searchInput = document.getElementById("symptom-search");
	const categorySelect = document.getElementById("category-select");
	const categories = document.querySelectorAll(".symptom-category");
	const symptomItems = document.querySelectorAll(".symptom-item");

	if (!searchInput || !categorySelect || categories.length === 0 || symptomItems.length === 0) {
		return;
	}

	categories.forEach((category) => {
		category.dataset.initialOpen = category.hasAttribute("open")
			? "true"
			: "false";
	});

	searchInput.addEventListener("input", (event) => {
		const query = event.target.value.trim().toLowerCase();

		categories.forEach((category) => {
			if (category.style.display === "none") {
				return;
			}
			const items = category.querySelectorAll(".symptom-item");
			let visibleCount = 0;

			items.forEach((item) => {
				const name = item.dataset.name || "";
				const isMatch = query === "" ? true : name.includes(query);
				item.style.display = isMatch ? "flex" : "none";
				if (isMatch) {
					visibleCount += 1;
				}
			});

			category.style.display = visibleCount > 0 ? "block" : "none";
			if (query === "") {
				category.open = category.dataset.initialOpen === "true";
			} else if (visibleCount > 0) {
				category.open = true;
			}
		});
	});

	categorySelect.addEventListener("change", (event) => {
		const selected = event.target.value;

		categories.forEach((category) => {
			const categoryKey = category.dataset.category || "";
			const shouldShow = categoryKey === selected;
			category.style.display = shouldShow ? "block" : "none";
			if (shouldShow) {
				category.open = true;
			}
		});

		searchInput.dispatchEvent(new Event("input"));
	});

	categories.forEach((category) => {
		category.style.display = "none";
	});
});
