% TOOL TAXONOMY LOGIC — ancestry and subsumption for abstract tool classes
% Docs: DOCS/encoding/tool_taxonomy_logic.md

% --- Abstract class and concrete tool detection ---
tool_abstract_class(Class) :- tool_taxonomy(_, (_, Class)).

tool_concrete(Tool) :-
    tool(Tool),
    not tool_abstract_class(Tool).

% --- Direct parent relationship ---
tool_parent(Child, Parent) :- tool_taxonomy(_, (Child, Parent)).

% --- Transitive ancestry ---
tool_ancestor(Child, Parent) :-
    tool_parent(Child, Parent).

tool_ancestor(Child, Ancestor) :-
    tool_parent(Child, Mid),
    tool_ancestor(Mid, Ancestor).

% --- Subsumption: tool_is_a(X, Y) if X is Y or a descendant of Y ---
tool_is_a(Tool, Tool) :- tool(Tool).

tool_is_a(Tool, Class) :-
    tool(Tool),
    tool_ancestor(Tool, Class).

tool_is_a(Class, Ancestor) :-
    tool_abstract_class(Class),
    tool_ancestor(Class, Ancestor).

tool_is_a(Class, Class) :-
    tool_abstract_class(Class).

tool_direct_selector(Tool, Tool) :-
    tool(Tool).

tool_direct_selector(Tool, Selector) :-
    tool(Tool),
    tool_taxonomy(_, (Tool, Selector)).

constraint_selector_matches_tool(Sel, Tool) :-
    constraint_selector(Sel, Selector),
    constraint_selector_mode(Sel, "transitive"),
    tool_is_a(Tool, Selector).

constraint_selector_matches_tool(Sel, Tool) :-
    constraint_selector(Sel, Selector),
    constraint_selector_mode(Sel, "exact"),
    tool_direct_selector(Tool, Selector).

% --- Class membership ---
tools_in_class(Class, Tool) :-
    tool_is_a(Tool, Class),
    tool(Tool).

% --- Development aid: tools not in any taxonomy class ---
% Uncomment to enable:
% tool_not_in_taxonomy(Tool) :-
%     tool(Tool),
%     not tool_ancestor(Tool, _).
