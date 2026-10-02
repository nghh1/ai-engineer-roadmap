// Functions
// Parameters, default, optional
function train(epochs: number, lr = 0.001, verbose?: boolean): void {}
// Arrow function with typed return
const add = (a: number, b: number): number => a+b;
// Function type
type Transform = (x: number) => number;
const relu: Transform = (x) => Math.max(0, x);
// Object parameter
interface GenerateOptions {
    prompt: string;
    temperature: number;
    maxTokens: number;
}
function generate({prompt, temperature=0.1, maxTokens=512}: GenerateOptions): void {}
// Rest params
function sum(...nums: number[]) {return nums.reduce((a, b) => a+b, 0);}
// Function overload - different return types for different inputs
function embed(input: string): Promise<number[]>;
function embed(input: string[]): Promise<number[][]>;
async function embed(input: string | string[]): Promise<number[] | number[][]> {
    return [];
}
// Type guard
function isString(x: unknown): x is string {
    return typeof x == 'string';
}

// Generics - one function/type that works over many types while keeping them connected
function first<T>(arr: T[]): T | undefined {
    return arr[0];
}
first(['a', 'b']); // returns string | undefined
first([1, 2]); // return number | undefined
// Constraints
function longest<T extends {length: number}>(a: T, b: T): T {
    return a.length >= b.length ? a : b;
}
// keyof constraint
function pluck<T, K extends keyof T>(items: T[], key: K): T[K][] {
    return items.map((i) => i[key]);
}
// Generic interface - classic API response wrapper
interface ApiResponse<T> {
    data: T;
    error: string | null;
    meta: {page: number; total: number};
}
// Generic typed fetch helper
async function getJSON<T>(url: string): Promise<T> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json() as Promise<T>;
}
// Generic class - in-memory vector store
class VectorStore<TMeta> {
    private items: {id: string; vector: number[]; meta: TMeta}[] = [];
    add(id: string, vector: number[], meta: TMeta) {
        this.items.push({id, vector, meta});
    }
    all() {return this.items;}
}
const store = new VectorStore<{source: string; page: number}>();