//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- Cypher AST types
// 1:1 port of uqa/graph/cypher/ast.py

// -- Expression types ---------------------------------------------------------

export interface LiteralExpr {
  readonly kind: "literal";
  readonly value: string | number | boolean | null;
}

export interface IdentifierExpr {
  readonly kind: "identifier";
  readonly name: string;
}

export interface PropertyAccessExpr {
  readonly kind: "property_access";
  readonly object: CypherExpr;
  readonly property: string;
}

export interface ParameterExpr {
  readonly kind: "parameter";
  readonly name: string;
}

export interface BinaryOpExpr {
  readonly kind: "binary_op";
  readonly op: string;
  readonly left: CypherExpr;
  readonly right: CypherExpr;
}

export interface UnaryOpExpr {
  readonly kind: "unary_op";
  readonly op: string;
  readonly operand: CypherExpr;
}

export interface FunctionCallExpr {
  readonly kind: "function_call";
  readonly name: string;
  readonly args: CypherExpr[];
}

export interface ListExpr {
  readonly kind: "list";
  readonly elements: CypherExpr[];
}

export interface InExpr {
  readonly kind: "in";
  readonly value: CypherExpr;
  readonly list: CypherExpr;
}

export interface IsNullExpr {
  readonly kind: "is_null";
  readonly value: CypherExpr;
  readonly negated: boolean;
}

export interface CaseExpr {
  readonly kind: "case";
  readonly operand: CypherExpr | null;
  readonly whens: Array<{ when: CypherExpr; then: CypherExpr }>;
  readonly elseExpr: CypherExpr | null;
}

export interface ExistsExpr {
  readonly kind: "exists";
  readonly pattern: NodePattern;
}

export interface IndexExpr {
  readonly kind: "index";
  readonly value: CypherExpr;
  readonly index: CypherExpr;
}

export type CypherExpr =
  | LiteralExpr
  | IdentifierExpr
  | PropertyAccessExpr
  | ParameterExpr
  | BinaryOpExpr
  | UnaryOpExpr
  | FunctionCallExpr
  | ListExpr
  | InExpr
  | IsNullExpr
  | CaseExpr
  | ExistsExpr
  | IndexExpr;

// -- Pattern types ------------------------------------------------------------

export interface NodePattern {
  readonly variable: string | null;
  readonly labels: string[];
  readonly properties: Map<string, CypherExpr>;
}

export interface RelationshipPattern {
  readonly variable: string | null;
  readonly types: string[];
  readonly properties: Map<string, CypherExpr>;
  readonly direction: "out" | "in" | "both";
  readonly minHops: number | null;
  readonly maxHops: number | null;
}

export interface PathPattern {
  readonly elements: Array<NodePattern | RelationshipPattern>;
}

// -- Clause types -------------------------------------------------------------

export interface MatchClause {
  readonly kind: "match";
  readonly patterns: PathPattern[];
  readonly optional: boolean;
  readonly where: CypherExpr | null;
}

export interface ReturnClause {
  readonly kind: "return";
  readonly items: ReturnItem[];
  readonly distinct: boolean;
  readonly orderBy: OrderByItem[] | null;
  readonly skip: CypherExpr | null;
  readonly limit: CypherExpr | null;
}

export interface ReturnItem {
  readonly expr: CypherExpr;
  readonly alias: string | null;
}

export interface OrderByItem {
  readonly expr: CypherExpr;
  readonly ascending: boolean;
}

export interface WithClause {
  readonly kind: "with";
  readonly items: ReturnItem[];
  readonly distinct: boolean;
  readonly where: CypherExpr | null;
  readonly orderBy: OrderByItem[] | null;
  readonly skip: CypherExpr | null;
  readonly limit: CypherExpr | null;
}

export interface CreateClause {
  readonly kind: "create";
  readonly patterns: PathPattern[];
}

export interface DeleteClause {
  readonly kind: "delete";
  readonly exprs: CypherExpr[];
  readonly detach: boolean;
}

export interface SetClause {
  readonly kind: "set";
  readonly items: SetItem[];
}

export interface SetItem {
  readonly property: PropertyAccessExpr;
  readonly value: CypherExpr;
  readonly operator?: string;
}

export interface RemoveClause {
  readonly kind: "remove";
  readonly items: CypherExpr[];
}

export interface MergeClause {
  readonly kind: "merge";
  readonly pattern: PathPattern;
  readonly onCreate: SetClause | null;
  readonly onMatch: SetClause | null;
}

export interface UnwindClause {
  readonly kind: "unwind";
  readonly expr: CypherExpr;
  readonly alias: string;
}

export interface CallClause {
  readonly kind: "call";
  readonly procedure: string;
  readonly args: CypherExpr[];
  readonly yields: string[] | null;
}

export type CypherClause =
  | MatchClause
  | ReturnClause
  | WithClause
  | CreateClause
  | DeleteClause
  | SetClause
  | RemoveClause
  | MergeClause
  | UnwindClause
  | CallClause;

// -- Top-level query ----------------------------------------------------------

export interface CypherQuery {
  readonly clauses: CypherClause[];
}

// -- Factory helpers ----------------------------------------------------------

export function createLiteral(value: string | number | boolean | null): LiteralExpr {
  return { kind: "literal", value };
}

export function createIdentifier(name: string): IdentifierExpr {
  return { kind: "identifier", name };
}

export function createPropertyAccess(
  object: CypherExpr,
  property: string,
): PropertyAccessExpr {
  return { kind: "property_access", object, property };
}

export function createParameter(name: string): ParameterExpr {
  return { kind: "parameter", name };
}

export function createBinaryOp(
  op: string,
  left: CypherExpr,
  right: CypherExpr,
): BinaryOpExpr {
  return { kind: "binary_op", op, left, right };
}

export function createUnaryOp(op: string, operand: CypherExpr): UnaryOpExpr {
  return { kind: "unary_op", op, operand };
}

export function createFunctionCall(name: string, args: CypherExpr[]): FunctionCallExpr {
  return { kind: "function_call", name, args };
}

export function createNodePattern(
  variable: string | null,
  labels?: string[],
  properties?: Map<string, CypherExpr>,
): NodePattern {
  return {
    variable,
    labels: labels ?? [],
    properties: properties ?? new Map<string, CypherExpr>(),
  };
}

export function createRelationshipPattern(
  opts?: Partial<{
    variable: string | null;
    types: string[];
    properties: Map<string, CypherExpr>;
    direction: "out" | "in" | "both";
    minHops: number | null;
    maxHops: number | null;
  }>,
): RelationshipPattern {
  return {
    variable: opts?.variable ?? null,
    types: opts?.types ?? [],
    properties: opts?.properties ?? new Map<string, CypherExpr>(),
    direction: opts?.direction ?? "out",
    minHops: opts?.minHops ?? null,
    maxHops: opts?.maxHops ?? null,
  };
}

export function createPathPattern(
  elements: Array<NodePattern | RelationshipPattern>,
): PathPattern {
  return { elements };
}

export function createCypherQuery(clauses: CypherClause[]): CypherQuery {
  return { clauses };
}
