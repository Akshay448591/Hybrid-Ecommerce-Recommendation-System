import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class CatalogService:
    def __init__(self, data_path: Optional[Path] = None, personas_path: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent.parent / "data"
        if data_path is None:
            data_path = base_dir / "curated_catalog.json"
        if personas_path is None:
            personas_path = base_dir / "sample_personas.json"
            
        self.catalog: List[Dict[str, Any]] = json.loads(data_path.read_text())
        self.personas: List[Dict[str, Any]] = json.loads(personas_path.read_text())
        self.item_by_id: Dict[int, Dict[str, Any]] = {item["id"]: item for item in self.catalog}

    def get_all(
        self,
        category: Optional[str] = None,
        department: Optional[str] = None,
        query: Optional[str] = None,
        sort_by: Optional[str] = None,  # 'price_asc', 'price_desc', 'rating', 'popular'
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        results = self.catalog.copy()

        if category and category.lower() != "all":
            results = [x for x in results if x["category"].lower() == category.lower()]

        if department and department.lower() != "all":
            results = [x for x in results if x["department"].lower() == department.lower()]

        if query:
            q = query.lower().strip()
            results = [
                x for x in results
                if q in x["title"].lower()
                or q in x["brand"].lower()
                or q in x["description"].lower()
                or any(q in t for t in x.get("tags", []))
            ]

        # Sorting
        if sort_by == "price_asc":
            results.sort(key=lambda x: x["price"])
        elif sort_by == "price_desc":
            results.sort(key=lambda x: x["price"], reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda x: x["rating"], reverse=True)
        elif sort_by == "popular":
            results.sort(key=lambda x: x["reviews_count"], reverse=True)

        total = len(results)
        paged = results[offset : offset + limit]

        return {
            "total": total,
            "items": paged,
            "categories": list(dict.fromkeys(x["category"] for x in self.catalog)),
            "departments": list(dict.fromkeys(x["department"] for x in self.catalog))
        }

    def get_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        return self.item_by_id.get(item_id)

    def get_personas(self) -> List[Dict[str, Any]]:
        return self.personas

    def get_persona(self, user_id: int) -> Optional[Dict[str, Any]]:
        for p in self.personas:
            if p["user_id"] == user_id:
                return p
        return None
