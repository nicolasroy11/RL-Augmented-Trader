import astroid
from termcolor import colored
from constants import TAB, CR
import helpers

class CodeGen:
    # gets run once per enum declaration
    def generate_client_constants(enumClass: astroid.ClassDef) -> str:
        body_string = 'export enum ' + enumClass.name + ' {' + CR

        for i, field in enumerate(enumClass.body):
            if not isinstance(field.value, astroid.Const):
                print(colored(f"Enums not of type [str: const] are not supported at the moment. Skipping '{enumClass.name}' -NR.", 'yellow'))
                return ''

            # we collect all the dto classes and interfaces
            # this will be used by the endpoints generator 
            # to import all the necessary DTOs
            helpers.add_to_imported_enums(enumClass.name)

            # enum field
            body_string += TAB + field.targets[0].name
            body_string += ' = '
            body_string += '\"' + str(field.value.value) + '\"'
            if (i != len(enumClass.body) - 1):
                body_string += ','

            body_string += CR

        # close body string
        body_string += '}' + CR + CR

        return body_string