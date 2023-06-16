from __future__ import annotations
from typing import Any, Iterator

from jax.tree_util import (
    PyTreeDef, tree_flatten, treedef_children, treedef_is_leaf,
    tree_unflatten
)


# TODO: Implement simpler version of swapping tree-branches.
# This can be done with tree-map and implementing a `is_leaf` callable.

def recurse_tree(tree: Any, stop_branch: PyTreeDef | None) -> Iterator[Any]:
    """Recursively traverse a PyTree from top-to-bottom and yield its leaves

    This yields the same data as tree_leaves(tree) aside from the fact
    that stop_branch cuts the tree recursion short when encountered within
    the (sub-)children of tree.

    Parameters
    ----------
    tree PyTree
        The datastructure to recursively traverse
    stop_branch PyTreeDef | None
        If not None, the tree-structure definition to cut recursion short

    Yields
    ------
    Any
        Leaf objects of tree

    Raises
    ------
    ValueError
        Thrown when the stop_branch is never encountered (if not None)
    """
    leaves, treedef = tree_flatten(tree)
    children_stack, data_gen = treedef_children(treedef)[::-1], iter(leaves)

    try:
        while branch := children_stack.pop():
            if (stop_branch is not None) and (branch == stop_branch):
                return

            if treedef_is_leaf(branch):
                yield next(data_gen)
            else:
                children_stack += treedef_children(branch)[::-1]
    except IndexError as e:
        if stop_branch is not None:
            raise ValueError(
                f"Stop-condition not met! {stop_branch} not in {treedef}"
            ) from e


def swap_branches(tree: Any, sub_tree: Any, suppress: bool = False) -> Any:
    """Swap an internal PyTree within a larger PyTree if it matches

    This function does not check for multiple encounters of sub_tree within
    tree and greedily swaps for the first encountered sub-tree in left-to-
    right order (in string representation/ tree_leaves(tree) order).

    Swapping multiple branches can instead be achieved by mapping this
    function over sub-trees of tree.

    Parameters
    ----------
    tree PyTree
        Some arbitrary data-structure containing values
    sub_tree PyTree
        Some data-structure whose TreeDef is a subset of `tree`
    suppress bool
        Whether to suppress the ValueError if sub_tree is not found in tree

    Returns
    -------
    Any
        Pytree of the same structure as tree with the swapped sub_tree

    Raises
    ------
    ValueError
        Chained if sub_tree isn't contained in tree and suppress is False
    """
    leaves, treedef = tree_flatten(tree)
    sub_leaves, subdef = tree_flatten(sub_tree)

    try:
        truncated_leaves = list(recurse_tree(tree, stop_branch=subdef))
    except ValueError:
        if not suppress:
            raise
    else:
        new_leaves = truncated_leaves + sub_leaves
        new_leaves += leaves[len(new_leaves):]
        tree = tree_unflatten(treedef, new_leaves)

    return tree
