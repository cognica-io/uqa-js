//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- RPQ optimizer
// 1:1 port of uqa/graph/rpq_optimizer.py

import { Alternation, Concat, KleeneStar, type RegularPathExpr } from "./pattern.js";

// -- Algebraic simplification -------------------------------------------------

export function simplifyExpr(expr: RegularPathExpr): RegularPathExpr {
  if (expr instanceof KleeneStar) {
    const inner = simplifyExpr(expr.inner);
    // (a*)* -> a*
    if (inner instanceof KleeneStar) {
      return inner;
    }
    return new KleeneStar(inner);
  }

  if (expr instanceof Alternation) {
    const left = simplifyExpr(expr.left);
    const right = simplifyExpr(expr.right);
    // a | a -> a
    if (left.equals(right)) {
      return left;
    }
    // a* | a -> a*
    if (left instanceof KleeneStar && left.inner.equals(right)) {
      return left;
    }
    // a | a* -> a*
    if (right instanceof KleeneStar && right.inner.equals(left)) {
      return right;
    }
    return new Alternation(left, right);
  }

  if (expr instanceof Concat) {
    const left = simplifyExpr(expr.left);
    const right = simplifyExpr(expr.right);
    // a* / a* -> a*
    if (
      left instanceof KleeneStar &&
      right instanceof KleeneStar &&
      left.inner.equals(right.inner)
    ) {
      return left;
    }
    return new Concat(left, right);
  }

  return expr;
}

// -- NFA types (shared with operators) ----------------------------------------

export interface NFAState {
  readonly id: number;
}

export interface NFATransition {
  readonly from: number;
  readonly to: number;
  readonly label: string | null; // null = epsilon
}

export interface NFA {
  readonly states: NFAState[];
  readonly transitions: NFATransition[];
  readonly startState: number;
  readonly acceptState: number;
}

// -- DFA types ----------------------------------------------------------------

// A DFA state is a serialized frozenset of NFA state ids
export type DFAState = string;
export type DFATransitions = Map<string, Map<string, string>>;

export interface DFAResult {
  readonly transitions: DFATransitions;
  readonly startState: DFAState;
  readonly acceptStates: Set<DFAState>;
}

// -- Subset construction ------------------------------------------------------

function serializeStateSet(states: Set<number>): DFAState {
  return [...states].sort((a, b) => a - b).join(",");
}

function epsilonClosure(nfa: NFA, states: Set<number>): Set<number> {
  const closure = new Set(states);
  const stack = [...states];
  while (stack.length > 0) {
    const current = stack.pop()!;
    for (const t of nfa.transitions) {
      if (t.from === current && t.label === null && !closure.has(t.to)) {
        closure.add(t.to);
        stack.push(t.to);
      }
    }
  }
  return closure;
}

function move(nfa: NFA, states: Set<number>, label: string): Set<number> {
  const result = new Set<number>();
  for (const t of nfa.transitions) {
    if (states.has(t.from) && t.label === label) {
      result.add(t.to);
    }
  }
  return result;
}

function getAlphabet(nfa: NFA): Set<string> {
  const alphabet = new Set<string>();
  for (const t of nfa.transitions) {
    if (t.label !== null) {
      alphabet.add(t.label);
    }
  }
  return alphabet;
}

export function subsetConstruction(nfa: NFA): DFAResult {
  const alphabet = getAlphabet(nfa);
  const startClosure = epsilonClosure(nfa, new Set([nfa.startState]));
  const startState = serializeStateSet(startClosure);

  const transitions: DFATransitions = new Map();
  const acceptStates = new Set<DFAState>();

  // Map from serialized state set to the actual state set
  const unmarked: Array<[DFAState, Set<number>]> = [[startState, startClosure]];
  const marked = new Set<DFAState>();

  if (startClosure.has(nfa.acceptState)) {
    acceptStates.add(startState);
  }

  while (unmarked.length > 0) {
    const [dfaState, nfaStates] = unmarked.pop()!;
    if (marked.has(dfaState)) continue;
    marked.add(dfaState);

    for (const symbol of alphabet) {
      const moveResult = move(nfa, nfaStates, symbol);
      if (moveResult.size === 0) continue;

      const closure = epsilonClosure(nfa, moveResult);
      const targetState = serializeStateSet(closure);

      let stateTransitions = transitions.get(dfaState);
      if (!stateTransitions) {
        stateTransitions = new Map();
        transitions.set(dfaState, stateTransitions);
      }
      stateTransitions.set(symbol, targetState);

      if (closure.has(nfa.acceptState)) {
        acceptStates.add(targetState);
      }

      if (!marked.has(targetState)) {
        unmarked.push([targetState, closure]);
      }
    }
  }

  return { transitions, startState, acceptStates };
}
