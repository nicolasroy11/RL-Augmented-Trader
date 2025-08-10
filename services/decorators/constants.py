from typing import Dict


TYPE_MAP: Dict[str, str] = {
    "bool": "boolean",
    "str": "string",
    "int": "number",
    "float": "number",
    "complex": "number",
    "Any": "any",
    "List": "any[]",
    "Literal": "\'any\'",
    "Tuple": "[any]",
    "Union": "any | any",
    "uuid": "string",
    "UUID": "string",
    "datetime": "string",
    "datetime.datetime": "string",
    "timezone": "string"
}

SUBSCRIPT_FORMAT_MAP: Dict[str, str] = {
    "List": "%s[]",
    "Literal": "\'%s\'",
    "Optional": "%s",
    "Tuple": "[%s]",
    "Union": "%s | %s",
    "Dict": "{ [key: string]: %s; }"
}

TAB = '\t'
CR = '\n'