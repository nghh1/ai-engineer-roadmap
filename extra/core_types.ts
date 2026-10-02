// Primitives
let characterName: string = 'ada';
let epoch: number = 10;
let isTraining: boolean = true;
let big: bigint = 10n;
// Let TS figures out which type it should be when it's obvious
let lr = 0.0001;
// Arrays and tuples
const losses: number[] = [0.9, 0.5, 0.3];
const labels: Array<string> = ["cat", "dog"];
const shape: [number, number] = [28, 28]; // tuple with fixed types
const point: [x: number, y: number] = [1, 2]; // labelled tuple
// Objects
const user: {id: number; email: string; nickname?: string} = {
    id: 1,
    email: "abcd@email.com",
    // nickname is optional
};
// Read only
const config: Readonly<{apiUrl: string}> = {apiUrl: "/api"};
const dims: readonly number[] = [1, 2, 3];
// Special types
let anything: any; // turns Off type checking
let mystery: unknown; // a safe ver. of the type any
function fail(msg: string): never {throw new Error(msg);}
function log(msg: string): void {console.log(msg);}
// undefined / null (separate types in strict mode)
let maybe: string | null = null;
/*
Rule of thumb: use unknown type instead of any for data don't trust yet
(API responses, JSON.parse, LLM output)
*/