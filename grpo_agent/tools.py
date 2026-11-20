"""The agent's tool: a safe calculator.

The model calls it by writing  <calc>EXPRESSION</calc>.  The expression is parsed with
Python's `ast` module and only numbers and + - * / // % ** ( ) are allowed, so the model
can never run arbitrary code.
"""
import ast
import math
import operator

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_ABS = 1e15


class ToolError(ValueError):
    pass


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ToolError("exponent too large")
        value = _BIN[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        value = _UNARY[type(node.op)](_eval(node.operand))
    else:
        raise ToolError("unsupported expression")
    if isinstance(value, complex) or not math.isfinite(value) or abs(value) > MAX_ABS:
        raise ToolError("result out of range")
    return value


def format_number(x):
    """Print integers without '.0' and round floats to 6 decimals."""
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if isinstance(x, float):
        return ("%.6f" % x).rstrip("0").rstrip(".")
    return str(x)


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression. Returns the result, or 'error: ...'."""
    expr = expression.strip().replace(",", "").replace("×", "*").replace("÷", "/").replace("^", "**").replace("$", "")
    if not expr or len(expr) > 200:
        return "error: empty or too long"
    try:
        return format_number(_eval(ast.parse(expr, mode="eval")))
    except ZeroDivisionError:
        return "error: division by zero"
    except (ToolError, SyntaxError, TypeError, ValueError, OverflowError) as e:
        return "error: %s" % (e if isinstance(e, ToolError) else "invalid expression")
