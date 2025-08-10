from collections import deque
from typing import List

from astroid.scoped_nodes import ClassDef
from .generator import CodeGen
import argparse, astroid
import os
import codegen_settings as settings
import helpers

def main(files):
    
    list_of_dtos: List[str] = get_all_dtos_in_files(files)
    list_of_enums: List[str] = get_all_enums_in_files(files)
    for file in files:
        
        # text representation of the code as read from the file path supplied
        code_as_text: str = open(file, "r").read()
        
        # generates a representation of the code as a tree of nodes
        code_nodes = astroid.parse(code_as_text)

        queue = deque([code_nodes])

        # navigate through the nodes and...
        while queue:
            current = queue.popleft()
            children = current.get_children()

            # ... ignore nodes that aren't a class and ...
            if not isinstance(current, astroid.ClassDef):
                queue.extend(children)
                continue
            
            # ... look for the one with the @View decorator.
            # Once you find it, you give it to the code generator.
            # we break, because we assume one class per file
            # if that changes, remove the break statement
            if helpers.has_decorator_with_name("View", current.decorators):
                generate(current, list_of_dtos, list_of_enums)
                break

# get the file content string and push it to the generated folder
def generate(view: ClassDef, list_of_dtos: List[str], list_of_enums: List[str]):
    if not os.path.exists(settings.ENDPOINTS_OUTDIR):
        os.makedirs(settings.ENDPOINTS_OUTDIR)
    file_content = CodeGen.generate_client_view_api(view, list_of_dtos, list_of_enums)
    f = open(settings.ENDPOINTS_OUTDIR + '/' + view.name.lower() + "-client" + ".ts", "w")   # 'r' for reading and 'w' for writing
    f.write(file_content)
    f.close()

# get the sh command arguments, namely the file name(s)
def get_args_namespace() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="Generates TypeScript HTTP client classes from views with the View decorator.")
    argparser.add_argument("path", action="store")
    # argparser.add_argument("-o, --outpath", action="store", default="api-interface.ts", dest="outpath")
    return argparser.parse_args()

def get_all_dtos_in_files(files) -> List[str]:
    list_of_dtos: List[str] = []
    for file in files:
        code_as_text: str = open(file, "r").read()
        
        # generates a representation of the code as a tree of nodes
        code_nodes = astroid.parse(code_as_text)

        queue = deque([code_nodes])

        while queue:
            current = queue.popleft()
            children = current.get_children()

            # ... ignore nodes that aren't a class and ...
            if not isinstance(current, astroid.ClassDef):
                queue.extend(children)
                continue
            
            if helpers.has_decorator_with_name("Dto", current.decorators):
                    list_of_dtos.append(current.name)
    return list_of_dtos

def get_all_enums_in_files(files) -> List[str]:
    list_of_enums: List[str] = []
    for file in files:
        code_as_text: str = open(file, "r").read()
        
        # generates a representation of the code as a tree of nodes
        code_nodes = astroid.parse(code_as_text)

        queue = deque([code_nodes])

        while queue:
            current = queue.popleft()
            children = current.get_children()

            # ... ignore nodes that aren't a class and ...
            if not isinstance(current, astroid.ClassDef):
                queue.extend(children)
                continue
            
            if helpers.has_decorator_with_name("cgEnum", current.decorators):
                    list_of_enums.append(current.name)
    return list_of_enums

if __name__ == "__main__":
    main()