from collections import deque
from typing import List
from .generator import CodeGen
import argparse, astroid
from termcolor import colored
import os
import helpers
import codegen_settings as settings

def main(files):
    file_content = ''
    enum_names: List[str] = []
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
            
            # ... look for the one with the @Enum decorator.
            # Once you find it, you give it to the code generator.
            # we break, because we assume one class per file
            # if that changes, remove the break statement
            if helpers.has_decorator_with_name("cgEnum", current.decorators):

                # avoid duplicates
                if current.name in enum_names:
                    print(colored(f"Enums not of type '{current.name}' has a duplicate; only keeping the first. Skipping '{current.name}' -NR.", 'yellow'))
                    continue
                enum_names.append(current.name)

                # create file content
                file_content += CodeGen.generate_client_constants(current)
                
            generate(file_content)

# get the file content string and push it to the generated file
def generate(file_content: str):
    if not os.path.exists(settings.ENUMS_OUTDIR):
        os.makedirs(settings.ENUMS_OUTDIR)

    f = open(settings.ENUMS_OUTDIR + "/enums" + ".ts", "w")   # 'r' for reading and 'w' for writing
    f.writelines(file_content)
    f.close()

# get the sh command arguments, namely the file name(s)
def get_args_namespace() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="Generates TypeScript enums.")
    # argparser.add_argument("path", action="store")
    # argparser.add_argument("-o, --outpath", action="store", default="api-interface.ts", dest="outpath")
    return argparser.parse_args()

if __name__ == "__main__":
    main()