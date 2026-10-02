// Interface and extension
interface User {
    id: string;
    name: string;
}
interface Admin extends User {
    permission: string[];
}

// type creation
type Role = 'user' | 'assistant' | 'system'; // unions: only `type` can do
type Vector = number[];
type WithTimestamps<T> = T & {createAt: Date; updateAt: Date;};
/**
 * Use interface for object shapes/structures extension (properties, API models).
 * Use type for unions, aliases, and computed types
 */

// Unions, Narrowing (refine a broad union type into a specific sub-type), and Discriminated Unions
/**
 * Keywords for narrowing: typeof, instanceof, in
 * Other narrowing techniques:
 * if (Array.isArray(x)){}
 * if (err instanceof Error) {}
 * if ("embedding" in obj) {}
 * if (value != null) {}
 */
function format(input: string | number) {
    if (typeof input === 'string')
        return input.toUpperCase();
    return input.toFixed(2);
}

/**
 * Discriminated Unions
 * Object unions where every constituent type shares a common literal property.
 */
type FetchState<T> = 
| {status: "idle"}
| {status: "loading"}
| {status: "success"; data: T}
| {status: "error"; error: string};
function render(state: FetchState<string[]>) {
    switch(state.status){
        case "idle":
            return "start";
        case "loading":
            return "Loading...";
        case "success":
            return state.data.join(", ");
        case "error":
            return state.error;
        default: {
            const _exhaustive: never = state;
            return _exhaustive;
        }
    }
}
