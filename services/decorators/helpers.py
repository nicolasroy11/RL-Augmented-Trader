# sifts through a list of astroid decorators and tells you whether one with decorator_name exists
from inspect import Attribute
from typing import List, Optional
import astroid
import constants

imported_dtos: List[str] = []
imported_enums: List[str] = []

def has_decorator_with_name(decorator_name: str, decorators: Optional[astroid.Decorators]) -> bool:
    if not decorators:
        return False

    return any(
        (getattr(decorator.func, "name", None) == decorator_name)
        if isinstance(decorator, astroid.Call)
        else decorator.name == decorator_name
        for decorator in decorators.nodes
    )


# sifts through a list of astroid decorators and returns the one with the given name
def get_decorator_by_name(decorator_name: str, decorators: astroid.Decorators):
    for decorator in decorators.nodes:
        if getattr(decorator.func, "name", None) == decorator_name:
            return decorator


# takes an astroid node and returns a TS types string
def get_ts_translation_from_node(node: astroid.node_classes.NodeNG) -> str:

    # first check if the value is a simple string name like 'str' or 'GridDto'...
    if isinstance(node, astroid.Name):

        # ... then if the type is in the primitive types TYPE_MAP dictionary ...
        if node.name in constants.TYPE_MAP.keys():
            type = constants.TYPE_MAP.get(node.name, node.name)
            return type

        # ... otherwise it's a simple custom type like 'GridDto' so just return that
        else: return node.name

    elif isinstance(node, astroid.Const) and node.name == "str":
        type = node.value
        return type

    elif isinstance(node, astroid.Subscript):
        subscript_value = node.value
        type_format = constants.SUBSCRIPT_FORMAT_MAP[subscript_value.name]
        if subscript_value.name == 'List':
            type = type_format % get_ts_translation_from_node(node.slice.value)
        elif subscript_value.name == 'Dict':
            type = type_format % get_ts_translation_from_node(node.slice.value.elts[1])
        elif subscript_value.name == 'Union':
            type = get_ts_translation_from_node(node.slice.value)
        elif subscript_value.name == 'Optional':
            type = type_format % get_ts_translation_from_node(node.slice.value)
        elif subscript_value.name == 'Literal':
            type = type_format % get_ts_translation_from_node(node.slice.value)
        return type

    elif isinstance(node, astroid.Tuple):
            inner_types = get_inner_tuple_types(node)
            delimiter = get_inner_tuple_delimiter(node)
            if delimiter != "UNKNOWN":
                type = delimiter.join(inner_types)
                return type

    elif isinstance(node, astroid.Call):
        call_value = node.func
        if call_value.attrname == 'Serializer':
            type = call_value.expr.name
            if node.keywords is not None and get_keyword_with_arg(node.keywords, 'many') is not None:
                type = constants.SUBSCRIPT_FORMAT_MAP['List'] % type
            return type

        type_format = constants.SUBSCRIPT_FORMAT_MAP[call_value.name]
        if call_value.name == 'Literal':
            type = type_format % node.args[0].value
            return type

    # as a default, return any
    return 'any'


def get_keyword_with_arg(keywords: List[astroid.Keyword], name: str) -> List[str]:
    child: astroid.Keyword
    for child in keywords:
        if child.arg is name:
            return child
    return None


def get_inner_tuple_types(tuple_node: astroid.Tuple) -> List[str]:
    # avoid using Set to keep order
    inner_types: List[str] = []
    for child in tuple_node.get_children():
        child_type = get_ts_translation_from_node(child)
        if child_type not in inner_types:
            inner_types.append(child_type)
    return inner_types


def get_inner_tuple_delimiter(tuple_node: astroid.Tuple) -> str:
    parent_subscript_name = tuple_node.parent.parent.value.name
    delimiter = "UNKNOWN"
    if parent_subscript_name == "Tuple":
        delimiter = ", "
    elif parent_subscript_name == "Union":
        delimiter = " | "
    return delimiter


def add_to_imported_dtos(type_name: str):
    if type_name not in imported_dtos:
        imported_dtos.append(type_name)

def add_to_imported_enums(type_name: str):
    if type_name not in imported_enums:
        imported_enums.append(type_name)
