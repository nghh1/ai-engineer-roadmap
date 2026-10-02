# Setup
## Initialise the project directory and create package.json, set "type": "module"
npm init -y
## Install TypeScript alongside the basic type definitions for Node.js
## Types are erased at runtime after compile, anything from input forms, storage, and models must be validated
npm install typescript @types/node ts-node zod
## Create a tsconfig.json 
npx tsc --init 
## React + TS frontend
npm create vite@latest app_name --template react-ts
## Full stack (frontend + API routes for AI apps)
npm create-next-app@latest app_name --typescript
# Dependencies and configurations
- package.json: This is the blueprint of your project. It lists your project's name, version, scripts, and exactly which external libraries (dependencies and devDependencies) your TypeScript code relies on. Without it, other developers cannot install the required packages.
- package-lock.json (or yarn.lock / pnpm-lock.yaml): This locks down the exact version of every single dependency and sub-dependency. It guarantees that anyone who clones your repository gets the exact same code environment, preventing breaking changes caused by silent third-party updates.
- tsconfig.json: This file contains your TypeScript compiler configuration (e.g., target JS version, strictness settings, module resolution). If you don't commit this, other developers and your deployment server won't know how to compile your .ts code into working JavaScript.
# Modules
```typescript
// In math.ts
export function dot(a: number[], b: number[]) {}
export type Vector = number[];
export class Matrix{}
// In main.ts
import Matrix, {dot} from './math';
import type {Vector} from './math';
```
