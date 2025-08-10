from termcolor import colored
import helpers
from decorators.dto_class_decorator import TsTypes 
from collections import deque
from typing import Any, Dict, List, NamedTuple
from constants import TAB, CR
import astroid


class Interface:
    pass


class PossibleInterfaceReference(str):
    pass


InterfaceAttributes = Dict[str, str]
PreparedInterfaces = Dict[str, InterfaceAttributes]

DtoDecorators = Dict[str, Any]


class Parser:
    def __init__(self, interface_qualname: str) -> None:
        self.interface_qualname = interface_qualname
        self.prepared: PreparedInterfaces = {}
        self.dto_decorators: DtoDecorators = {}

    def parse(self, code: str) -> None:
        queue = deque([astroid.parse(code)])

        while queue:
            current = queue.popleft()
            children = current.get_children()

            if not isinstance(current, astroid.ClassDef):
                queue.extend(children)
                continue

            if not helpers.has_decorator_with_name("Dto", current.decorators):
                # warnings.warn("Classes not decorated with @Dto are not supported -NR.", UserWarning)
                continue

            if current.name in self.prepared:
                print(colored(f"Type with name '{current.name}' seems to be duplicated; will keep the first only. -NR.", 'yellow'))
                continue
            
            self.dto_decorators[current.name] = helpers.get_decorator_by_name("Dto", current.decorators)
            self.prepared[current.name] = get_types_from_classdef(current)
        ensure_possible_interface_references_valid(self.prepared)

    # modified the original code to export the interfaces instead of merely declaring them - NR
    def flush(self) -> str:
        serialized: List[str] = []
        for class_name, attributes in self.prepared.items():

            # we collect all the dto classes and interfaces
            # this will be used by the endpoints generator 
            # to import all the necessary DTOs
            helpers.add_to_imported_dtos(class_name)

             # check if this will be a class or an interface (interface is default)
            ts_type = TsTypes.Interface
            if self.dto_decorators[class_name].keywords != None:
                keyword: astroid.Keyword
                for keyword in self.dto_decorators[class_name].keywords:
                    if keyword.arg == 'ts_type':
                        ts_type = TsTypes[keyword.value.attrname]

            attrs = attributes.items()

            string = f"export {ts_type.value} {class_name}" + " {" + CR

            # declare class members
            for name, attributes in attrs:
                string += TAB + name
                if attributes['is_optional'] == True: string += "?"
                string += f": {attributes['type']}"
                string += ";" + CR

            # include a constructor if this will be a class
            if ts_type == TsTypes.Class:
                string += CR
                string += TAB + 'constructor (' + CR
                for name, attributes in attrs:
                    string += TAB + TAB + name
                    if attributes['is_optional'] == True: string += "?"
                    string += f": {attributes['type']}"
                    # if attributes['default'] != None: string += f" = {attributes['default']}"  # if you ever need to set defaults
                    string += "," + CR
                string += TAB + ') {' + CR
                for name, attributes in attrs:
                    string += TAB + TAB + f"this.{name} = {name};" + CR
                string += TAB + '}' + CR

            string += "}" + CR
            serialized.append(string)

        self.prepared.clear()
        return CR.join(serialized)

def get_types_from_classdef(node: astroid.ClassDef) -> Dict[str, str]:
    serialized_types: Dict[str, str] = {}
    for child in node.body:

        # if isinstance(child, astroid.FunctionDef) and helpers.has_decorator_with_name('DtoField', child.decorators):
        #     child_name, child_type, child_default, is_child_optional = parse_function_node(child)
            
        if not isinstance(child, astroid.AnnAssign):
            continue 
        
        child_name, child_type, child_default, is_child_optional = parse_annassign_node(child)
        serialized_types[child_name] = dict( type = child_type, default = child_default, is_optional = is_child_optional )
    return serialized_types


class ParsedAnnAssign(NamedTuple):
    attr_name: str
    attr_type: str
    attr_default: Any
    is_attr_optional: bool


def parse_annassign_node(node: astroid.AnnAssign) -> ParsedAnnAssign:
    is_optional = check_if_optional(node)
    return ParsedAnnAssign(node.target.name, helpers.get_ts_translation_from_node(node.annotation), check_for_default_value(node), is_optional)

def parse_function_node(node: astroid.FunctionDef) -> ParsedAnnAssign:
    dto_decorator = helpers.get_decorator_by_name('DtoField', node.decorators)
    is_optional = False
    default_value = None
    if dto_decorator.keywords != None:
        keyword: astroid.Keyword
        for keyword in dto_decorator.keywords:
            if keyword.arg == 'is_optional':
                is_optional = keyword.value.value
            elif keyword.arg == 'default_value':
                pass
                # default_value = keyword.value
    return ParsedAnnAssign(node.name, helpers.get_ts_translation_from_node(node.returns), default_value, is_optional)

def check_for_default_value(node: astroid.AnnAssign) -> Any:
    if node.value != None:
        return node.value.value
    else: return None

def check_if_optional(node: astroid.AnnAssign) -> bool:
    if isinstance(node.annotation, astroid.Subscript):
        if node.annotation.value.name == 'Optional': return True
    return False

def ensure_possible_interface_references_valid(interfaces: PreparedInterfaces) -> None:
    interface_names = set(interfaces.keys())

    for interface, attributes in interfaces.items():
        for attribute_name, attribute_type in attributes.items():
            if not isinstance(attribute_type, PossibleInterfaceReference):
                continue

            if attribute_type not in interface_names:
                raise RuntimeError(
                    f"Invalid nested Interface reference '{attribute_type}'"
                    f" found for interface {interface}!\n"
                    f"Does '{attribute_type}' exist and is it an Interface?"
                )