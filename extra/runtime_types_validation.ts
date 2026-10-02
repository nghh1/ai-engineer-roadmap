import {z} from "zod";

const UserSchema = z.object({
    id: z.string(),
    email: z.email(),
    age: z.number().int().min(0).optional(),
    role: z.enum(['admin', 'member'])
});
// New type derived from schema
type User = z.infer<typeof UserSchema>;

async function fetchUser(id: string): Promise<User> {
    const res = await fetch(`/api/users/${id}`);
    const json: unknown = await res.json();
    return UserSchema.parse(json); // Unsuccessful parsing would throw away immediately
}
// Non-throwing version
const res = await fetch(`/api/users/001`);
const json_data: unknown = await res.json();
const result = UserSchema.safeParse(json_data);
if (result.success) {
    result.data.email;
}else{
    console.error(result.error.issues);
}
// Validating LLM structured output
const SentimentSchema = z.object({
    label: z.enum(["positive", "negative", "neutral"]),
    confidence: z.number().min(0).max(1),
    reasons: z.array(z.string())
});
function parseJSON(text: string) {
    const cleaned = text.replace(/```json|```/g, '').trim();
    try{
        return SentimentSchema.safeParse(JSON.parse(cleaned));
    }catch{
        return {success: false, error: "Invalid JSON"};
    }
}
// Utility types and type operators
/**
 * Awaited<Promise<string>> // unwrap the type within a Promise, useful for async functions
 * interface Product {id: string; name: string; price: number; tags: string[];}
 * Partial<Product> // use subset properties of the Product, great for patch or update forms
 * Required<Product> // all properties of the Product are strictly required
 * Readonly<Product> // all properties of the Product cannot be reassigned
 * Record<string, number> // constructs a dictionary {[key: string]: number}
 * Record<'train'|'val'|'test', number[]>
 * Pick<Product, 'id'|'name'> // pick specific subset of properties from the Product
 * Omit<Product, 'id'> // all properties except `id`, use to create payloads
 * Extract<'a'|'b'|'c', 'a'|'b'> // accept union type members 'a' or 'b' only
 * Exclude<'a'|'b'|'c', 'a'> // exclude union type member 'a'
 * NonNullable<string | null> // construct a type excluding null and undefined
 * ReturnType<typeof someFn> // construct a type that corresponds to the return type of a function
 * Parameters<typeof someFn> // build a tuple type from types used in parameters of a function
 * ConstructorParameters<type of someFn> // build a tuple/array type from types of a constructor
 */
const defaults = {temperature: 0.7, topP: 1};
type Defaults = typeof defaults; // number
type DefaultKey = keyof Defaults; // 'temperature' or 'topP'
type Temp = Defaults['temperature']; // number type assigned by index access
type Nullable<T> = {[K in keyof T]: T[K] | null}; // null or type is a mapping T[K] = value
type ElementOf<T> = T extends (infer U)[] ? U : never;
type N = ElementOf<number[]>; // type is same as the element type of a iterable object
type Env = 'dev' | 'prod';
type EnvVar = `API_URL_${Uppercase<Env>}`; // template literal types

type ModelConfig = Record<string, {contextWindow: number; costPer1k: number}>;
const models = {
    fast: {contextWindow: 200000, costPer1k: 0.001},
    smart: {contextWindow: 200000, costPer1k: 0.015}
} satisfies ModelConfig; // validate whether the assignment match the declaration of ModelConfig
models.fast; // access fast property and TS knows its shape by keyword `satisfies`
