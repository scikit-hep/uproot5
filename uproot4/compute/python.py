# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

import ast

import uproot4.compute


# _expression_is_branch = re.compile(
#     r"^\s*[A-Za-z_][A-Za-z_0-9]*(\s*\.\s*[A-Za-z_][A-Za-z_0-9]*)*\s*$"
# )
# _expression_strip_dot_whitespace = re.compile(r"\s*\.\s*")


def _expression_to_node(expression, file_path, object_path):
    try:
        node = ast.parse(expression)
    except SyntaxError as err:
        raise SyntaxError(
            err.args[0] + "\nin file {0} at {1}".format(file_path, object_path),
            err.args[1],
        )

    if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
        raise SyntaxError(
            "expected a single expression\nin file {0} at {1}".format(
                file_path, object_path
            )
        )

    return node


def _attribute_to_dotted_name(node):
    if isinstance(node, ast.Attribute):
        tmp = _attribute_to_dotted_name(node.value)
        if tmp is None:
            return None
        else:
            return tmp + "." + node.attr

    elif isinstance(node, ast.Name):
        return node.id

    else:
        return None


def _walk_ast_yield_symbols(node, aliases, functions):
    if isinstance(node, ast.Name):
        if node.id not in functions:
            yield node.id

    elif isinstance(node, ast.Attribute):
        name = _attribute_to_dotted_name(node)
        if name is None:
            for y in _walk_ast_yield_symbols(node.value, aliases, functions):
                yield y
        else:
            yield name

    elif isinstance(node, ast.AST):
        for field_name in node._fields:
            x = getattr(node, field_name)
            for y in _walk_ast_yield_symbols(x, aliases, functions):
                yield y

    elif isinstance(node, list):
        for x in node:
            for y in _walk_ast_yield_symbols(x, aliases, functions):
                yield y

    else:
        pass


def _ast_as_branch_expression(node, aliases, functions):
    if isinstance(node, ast.Name):
        if node.id in aliases:
            return ast.parse("get_alias({0})".format(repr(node.id))).body[0].value
        elif node.id in functions:
            return ast.parse("functions[{0}]".format(repr(node.id))).body[0].value
        else:
            return ast.parse("arrays[{0}]".format(repr(node.id))).body[0].value

    elif isinstance(node, ast.Attribute):
        name = _attribute_to_dotted_name(node)
        if name is None:
            value = _ast_as_branch_expression(node.value, aliases, functions)
            new_node = ast.Attribute(value, node.attr, node.ctx)
            new_node.lineno = getattr(node, "lineno", 1)
            new_node.col_offset = getattr(node, "col_offset", 0)
            return new_node
        else:
            return ast.parse("arrays[{0}]".format(repr(name))).body[0].value

    elif isinstance(node, ast.AST):
        args = []
        for field_name in node._fields:
            field_value = getattr(node, field_name)
            args.append(_ast_as_branch_expression(field_value, aliases, functions))
        new_node = type(node)(*args)
        new_node.lineno = getattr(node, "lineno", 1)
        new_node.col_offset = getattr(node, "col_offset", 0)
        return new_node

    elif isinstance(node, list):
        return [_ast_as_branch_expression(x, aliases, functions) for x in node]

    else:
        return node


def _expression_to_function(
    expression, aliases, functions, scope, file_path, object_path
):
    node = _expression_to_node(expression, file_path, object_path)
    expr = _ast_as_branch_expression(node.body[0].value, aliases, functions)
    function = ast.parse("lambda: None").body[0].value
    function.body = expr
    expression = ast.Expression(function)
    expression.lineno = getattr(function, "lineno", 1)
    expression.col_offset = getattr(function, "col_offset", 0)
    return eval(compile(expression, "<dynamic>", "eval"), scope)


class ComputePython(uproot4.compute.Compute):
    def __init__(self, functions=None):
        if functions is None:
            self._functions = {}
        else:
            self._functions = dict(functions)

    @property
    def functions(self):
        return self._functions

    def free_symbols(self, expression, aliases, file_path, object_path):
        node = _expression_to_node(expression, file_path, object_path)
        return _walk_ast_yield_symbols(node, aliases, self._functions)

    def compute_expressions(
        self, arrays, expression_context, aliases, file_path, object_path
    ):
        alias_values = {}

        def get_alias(alias_name):
            if alias_name not in alias_values:
                alias_values[alias_name] = _expression_to_function(
                    aliases[alias_name],
                    aliases,
                    self._functions,
                    scope,
                    file_path,
                    object_path,
                )()
            return alias_values[alias_name]

        scope = {"arrays": {}, "get_alias": get_alias, "functions": self._functions}
        for expression, context in expression_context:
            branch = context.get("branch")
            if branch is not None:
                scope["arrays"][expression] = arrays[id(branch)]

        output = {}
        for expression, context in expression_context:
            if context["is_primary"] and not context["is_cut"]:
                output[expression] = _expression_to_function(
                    expression, aliases, self._functions, scope, file_path, object_path,
                )()

        cut = None
        for expression, context in expression_context:
            if context["is_primary"] and context["is_cut"]:
                cut = _expression_to_function(
                    expression, aliases, self._functions, scope, file_path, object_path,
                )()
                break

        if cut is not None:
            cut = cut != 0
            for name in output:
                output[name] = output[name][cut]

        return output
