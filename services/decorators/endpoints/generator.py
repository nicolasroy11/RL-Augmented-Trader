from astroid.scoped_nodes import FunctionDef
import helpers
import re
import astroid
from typing import Dict, List
import codegen_settings as settings
from termcolor import colored
from constants import TAB, CR


class CodeGen:

    def generate_client_view_api(viewClass: astroid.ClassDef, list_of_dtos: List[str], list_of_enums: List[str]) -> str:

        def check_if_optional(node: astroid.AnnAssign) -> bool:
            if isinstance(node, astroid.Subscript):
                if node.value.name == 'Optional': return True
            return False

        header_string = 'import axios from "axios"; \n'
        body_string = 'export default class ' + \
            viewClass.name + "Client" + ' {\n\n'

        # get all methods from the view class
        methods_list = viewClass.body
        method_names: List[str] = []

        view_decorator = helpers.get_decorator_by_name("View", viewClass.decorators)
        view_ts_hash: Dict[str, str] = {}
        for keyword in view_decorator.keywords:
            view_ts_hash[keyword.arg] = keyword.value.value

        ######################## Handle each method with the View source ##################################################
        method: FunctionDef
        for i, method in enumerate(methods_list):

            # we ignore the methods that do not have the @Http decorator...
            if not helpers.has_decorator_with_name("Http", method.decorators):
                continue

            # ... or are not functions...
            if not isinstance(method, astroid.FunctionDef):
                print(colored(f"'{method.name}' in view '{viewClass.name}' is not a function definition. Only functions are supported at the moment. -NR.", 'yellow'))
                continue

            # ... or are duplicates
            if method.name in method_names:
                print(colored(f"Method with name '{method.name}' in view '{viewClass.name}' seems to be duplicated; will keep the first only. -NR.", 'yellow'))
                continue

            method_names.append(method.name)

            ################### prepare data obtained from HTTP decorator args #################################################

            http_decorator = helpers.get_decorator_by_name("Http", method.decorators)
            http_ts_hash: Dict[str, str] = {}

            # cycle through the supplied @Http decorator args and prepare
            # TS attributes for use in file generation
            keyword: astroid.Keyword
            for keyword in http_decorator.keywords:
                if keyword.arg == 'path':
                    path = keyword.value.value
                    path_ts = keyword.value.value
                    m = re.findall(r'\<.*?\>', path)
                    for i, s in enumerate(m):
                        url_var = '${' + re.sub('>', '', s.split(':')[1]) + '}'
                        path_ts = re.sub(s, url_var, path_ts)
                    http_ts_hash['path_ts'] = path_ts
                elif keyword.arg == 'return_type':
                    http_ts_hash['return_type_ts'] = helpers.get_ts_translation_from_node(keyword.value)
                elif keyword.arg == 'request_payload_type':
                    http_ts_hash['request_payload_type_ts'] = helpers.get_ts_translation_from_node(keyword.value)
                elif keyword.arg == 'query_params_type':
                    http_ts_hash['query_params_type_ts'] = helpers.get_ts_translation_from_node(keyword.value)
                elif keyword.arg == 'description':
                    pass
                else:
                    http_ts_hash[keyword.arg] = keyword.value.value


            ################### prepare TS method arguments from original py method arguments ################################
            
            method_args_hash: Dict[str, str] = {}
            optional_args_hash: Dict[str, str] = {}
            is_optional: Dict[str, bool] = {}

            # first check for the exec() function def inside the method
            # if there isn't one, we use the outside method
            if 'exec' in method.locals:
                _method = method.locals.get('exec')[0]
            else: _method = method
            for i, type in enumerate(_method.args.args):
                if isinstance(type, astroid.AssignName):
                    if type.name == 'self': continue

                    # check if the current argument is the same type as the payload, if it is, we assume this argument is the payload
                    if http_ts_hash.get('request_payload_type_ts', None) != None and _method.args.annotations[i].name == http_ts_hash['request_payload_type_ts']:
                        payload_var_name = type.name

                    method_args_hash[type.name] = helpers.get_ts_translation_from_node(_method.args.annotations[i])
                    is_optional[type.name] = check_if_optional(_method.args.annotations[i])

            ######################## Build method string ##################################################################

            # method signature
            body_string += TAB + 'public static async ' + method.name
            body_string += '('

            # method arguments
            for i, key in enumerate(method_args_hash):
                body_string += key
                if is_optional[key] == True: body_string += "?"
                body_string += ': '
                body_string += method_args_hash[key]
                if i != len(method_args_hash) - 1 or len(optional_args_hash) > 0:
                    body_string += ', '

            # return type
            body_string += '): Promise<'
            body_string += http_ts_hash['return_type_ts']
            body_string += '> {' + CR

            # url
            body_string += TAB + TAB + "let path: string = "
            body_string += f'`{settings.API_URL}:' + str(settings.API_PORT) + '/api/' + \
                view_ts_hash["url"] + http_ts_hash['path_ts'] + '`;' + CR
            
            # if there are optional params, this code will add them dynamically to the url as query params
            if "optional_params" in method_args_hash:
                body_string += TAB + TAB + "if (optional_params) {" + CR
                body_string += TAB + TAB + TAB + "path += '?'" + CR
                body_string += TAB + TAB + TAB + "var map: { [key: string]: any } = optional_params;" + CR
                body_string += TAB + TAB + TAB + "const urlParams: string[] = Object.keys(map).map(key => {" + CR
                body_string += TAB + TAB + TAB + TAB + "return key + \"=\" + map[key]" + CR
                body_string += TAB + TAB + TAB + "});" + CR
                body_string += TAB + TAB + TAB + "path += urlParams.join('&')" + CR
                body_string += TAB + TAB + "}" + CR

            # axios statement
            body_string += TAB + TAB + 'const req = await axios.'
            body_string += http_ts_hash['http_method'].lower()
            body_string += '(path'
            
            if http_ts_hash['http_method'].lower() == 'post':
                body_string += f', {payload_var_name}'
            body_string += ');' + CR

            # return statement
            body_string += TAB + TAB + 'return req.data;' + CR

            # method close
            body_string += TAB + '}' + CR + CR

        # close body string
        body_string += '}'

        # adding all imported dto dependencies to the head of the file
        imported_classes = helpers.imported_dtos
        if len(imported_classes) > 0:
            header_string += 'import { '
            for i, imported in enumerate(imported_classes):
                header_string += imported
                if (i != len(imported_classes) - 1):
                    header_string += ', '
            header_string += ' } from "./dtos";' + CR

        # adding all imported enum dependencies to the head of the file
        imported_enums = helpers.imported_enums
        if len(imported_enums) > 0:
            header_string += 'import { '
            for i, imported in enumerate(imported_enums):
                header_string += imported
                if (i != len(imported_enums) - 1):
                    header_string += ', '
            header_string += ' } from "./enums";' + CR

        header_string += CR

        return header_string + body_string