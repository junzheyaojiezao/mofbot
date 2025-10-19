import os, re, requests
from typing import Dict, Any, Optional, List
from urllib.parse import quote_plus

CROSSREF_BASE = "https://api.crossref.org/works"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
MP_BASE = "https://api.materialsproject.org/v2"

def crossref_search(query: str, rows: int = 5) -> Dict[str, Any]:
    query = query.strip()
    m = re.match(r'^(10\.\d{4,9}/\S+)$', query, flags=re.I)
    if m:
        doi = m.group(1)
        url = f"{CROSSREF_BASE}/{quote_plus(doi)}"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        item = r.json().get("message", {})
        return {"query": query, "items": [format_crossref_item(item)]}
    params = {"query": query, "rows": rows}
    r = requests.get(CROSSREF_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("message", {}).get("items", [])
    items = [format_crossref_item(x) for x in data]
    return {"query": query, "items": items}

def format_crossref_item(x: Dict[str, Any]) -> Dict[str, Any]:
    title = (x.get("title") or [""])[0]
    authors = []
    for a in x.get("author", [])[:8]:
        name = f"{a.get('given','').strip()} {a.get('family','').strip()}".strip()
        if name: authors.append(name)
    year = None
    if x.get("issued", {}).get("date-parts"):
        year = x["issued"]["date-parts"][0][0]
    doi = x.get("DOI")
    journal = (x.get("short-container-title") or x.get("container-title") or [""])[0]
    url = x.get("URL")
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "journal": journal,
        "url": url
    }

def pubchem_properties(name_or_cid: str) -> Dict[str, Any]:
    key = name_or_cid.strip()
    if re.fullmatch(r'\d+', key):
        path = f"compound/cid/{key}/property/MolecularFormula,MolecularWeight,IsomericSMILES,IUPACName/JSON"
    else:
        path = f"compound/name/{quote_plus(key)}/property/MolecularFormula,MolecularWeight,IsomericSMILES,IUPACName/JSON"
    url = f"{PUBCHEM_BASE}/{path}"
    r = requests.get(url, timeout=20)
    if r.status_code == 404:
        return {"query": key, "properties": []}
    r.raise_for_status()
    props = r.json().get("PropertyTable", {}).get("Properties", [])
    return {"query": key, "properties": props}

def mp_query(formula_or_mpid: str, api_key: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
    api_key = api_key or os.getenv("MATERIALS_PROJECT_API_KEY", "")
    headers = {"X-API-KEY": api_key} if api_key else {}
    q = formula_or_mpid.strip()
    if q.startswith("mp-"):
        url = f"{MP_BASE}/materials/{q}/summary"
    else:
        url = f"{MP_BASE}/materials/{quote_plus(q)}/summary"
    params = {"fields": "material_id,formula_pretty,chemical_system,energy_above_hull,band_gap,volume,structure", "limit": limit}
    r = requests.get(url, headers=headers, params=params, timeout=25)
    if r.status_code == 401:
        return {"error": "Materials Project API key missing or invalid", "query": q}
    if r.status_code == 404:
        return {"query": q, "results": []}
    r.raise_for_status()
    data = r.json().get("data", [])
    results = []
    for it in data[:limit]:
        results.append({
            "material_id": it.get("material_id"),
            "formula": it.get("formula_pretty"),
            "chemical_system": it.get("chemical_system"),
            "band_gap": it.get("band_gap"),
            "energy_above_hull": it.get("energy_above_hull"),
            "volume": it.get("volume"),
        })
    return {"query": q, "results": results}

def maybe_tool_call(question: str) -> Dict[str, Any]:
    q = question.strip()
    lower = q.lower()

    if lower.startswith("tool:crossref"):
        return {"crossref": crossref_search(q.split(" ", 1)[1] if " " in q else "")}
    if lower.startswith("tool:pubchem"):
        return {"pubchem": pubchem_properties(q.split(" ", 1)[1] if " " in q else "")}
    if lower.startswith("tool:mp"):
        return {"materials_project": mp_query(q.split(" ", 1)[1] if " " in q else "")}

    if "doi:" in lower or "crossref" in lower or "find paper" in lower or "paper" in lower:
        m = re.search(r'doi:\s*([\w./()-]+)', q, flags=re.I)
        if m:
            return {"crossref": crossref_search(m.group(1))}
        return {"crossref": crossref_search(q)}

    if "pubchem" in lower or "cid" in lower or "smiles" in lower or "molecular weight" in lower:
        m = re.search(r'(?:pubchem|cid)\s*[:=]?\s*([A-Za-z0-9\-\s]+)', q, flags=re.I)
        token = m.group(1).strip() if m else q
        return {"pubchem": pubchem_properties(token)}

    if "materials project" in lower or "mp-" in lower or looks_like_formula(q):
        token = extract_formula_or_mpid(q) or q
        return {"materials_project": mp_query(token)}

    return {}

def looks_like_formula(text: str) -> bool:
    import re as _re
    t = text.strip()
    if _re.search(r'\\bmp-\\d+\\b', t):
        return True
    if _re.search(r'\\b[A-Z][a-z]?(?:-[A-Z][a-z]?)+\\b', t):
        return True
    if _re.search(r'\\b([A-Z][a-z]?){1,3}\\d{0,3}\\b', t):
        return True
    return False

def extract_formula_or_mpid(text: str):
    import re as _re
    m = _re.search(r'\\b(mp-\\d+)\\b', text)
    if m:
        return m.group(1)
    m = _re.search(r'\\b([A-Z][a-z]?(?:-[A-Z][a-z]?)+)\\b', text)
    if m:
        return m.group(1)
    m = _re.search(r'\\b([A-Z][a-z]?\\d{0,3})\\b', text)
    if m:
        return m.group(1)
    return None
