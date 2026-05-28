from typing import Optional

from rdflib import URIRef


def _clean_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == '""':
        return None
    return text


def get_label(entity) -> str:
    labels = getattr(entity, "label", []) or getattr(entity, "prefLabel", []) or []
    if labels:
        cleaned = _clean_text(labels[0])
        if cleaned:
            return cleaned
    name = _clean_text(getattr(entity, "name", None))
    if name:
        return name
    iri = _clean_text(getattr(entity, "iri", None))
    if iri:
        return iri.split("#")[-1].rsplit("/", 1)[-1]
    fallback = _clean_text(str(entity))
    if fallback:
        return fallback
    return "Unnamed"


def get_comment(entity) -> Optional[str]:
    comments = getattr(entity, "comment", []) or []
    if isinstance(comments, str):
        return _clean_text(comments)
    if comments:
        return _clean_text(comments[0])
    return None


def get_definition(entity, include_pref_label: bool = False) -> str:
    definitions = []
    definitions.extend(getattr(entity, "IAO_0000115", []) or [])
    definitions.extend(getattr(entity, "definition", []) or [])

    comments = getattr(entity, "comment", []) or []
    if isinstance(comments, str):
        definitions.append(comments)
    else:
        definitions.extend(comments)

    if include_pref_label:
        definitions.extend(getattr(entity, "prefLabel", []) or [])

    world_obj = getattr(entity, "world", getattr(getattr(entity, "namespace", None), "world", None))
    if world_obj:
        try:
            for annotation_property in world_obj.annotation_properties():
                local_name = str(annotation_property.iri).split("#")[-1].rsplit("/", 1)[-1].lower()
                if "definition" not in local_name and "comment" not in local_name:
                    continue
                values = getattr(entity, annotation_property.python_name, []) or []
                if not isinstance(values, (list, tuple)):
                    values = [values]
                definitions.extend(values)
        except Exception:
            pass

        try:
            graph = world_obj.as_rdflib_graph()
            subject = URIRef(entity.iri)
            for predicate, obj in graph.predicate_objects(subject):
                local_name = str(predicate).split("#")[-1].rsplit("/", 1)[-1].lower()
                if "definition" in local_name:
                    definitions.append(obj)
        except Exception:
            pass

    english = next((_clean_text(value) for value in definitions if getattr(value, "lang", None) == "en"), None)
    if english:
        return english
    if definitions:
        text = _clean_text(definitions[0])
        if text:
            return text
    return "No definition provided."
