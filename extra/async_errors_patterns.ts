// Async, Errors, and useful patterns
// Parallel requests
async function fetchFromAPI(url: string){
    const res = await fetch(url);
    const res_json = await res.json();
    return JSON.parse(res_json);
}
const [users, posts] = await Promise.all([
    fetchFromAPI("/api/users"),
    fetchFromAPI("/api/posts")
]);
// Errors in catch are unknown
try{
    // await promise_ops();
}catch(error){
    const message = error instanceof Error ? error.message : String(error);
}
// Result type - explicit error handling instead of throwing
type Result<T, E = string> = {ok: true; value: T} | {ok: false; error: E};
async function safeFetch<T>(url: string): Promise<Result<T>> {
    try{
        const res = await fetch(url);
        if (!res.ok) return {ok: false, error: `HTTP ${res.status}`};
        return {ok: true, value: (await res.json()) as T};
    }catch(error){
        return {ok: false, error: error instanceof Error ? error.message : String(error)};
    }
}
// Retry with exponential backoff (common for LLM rate limits)
async function withRetry<T>(fn: () => Promise<T>, retries = 3, baseMs = 500): Promise<T> {
    for (let attempt=0; ; attempt++){
        try{
            return await fn();
        }catch(error){
            if (attempt >= retries) throw error;
            await new Promise((r) => setTimeout(r, baseMs**attempt));
        }
    }
}
// Timeouts / cancellation
const controller = new AbortController();
setTimeout(() => controller.abort(), 10000);
await fetch("/api/test", {signal: controller.signal});
// Branded types
type Brand<T, B> = T & {brand: B};
type UserID = Brand<string, "UserID">;
type DocID = Brand<string, "DocID">;
function getUser(id: UserID) {return id}
getUser("a12bc" as UserID);