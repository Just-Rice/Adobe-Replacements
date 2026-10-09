# Storyteller Studio

## Installing dependencies
* [Install Rust](https://www.rust-lang.org/tools/install)
* [Install Node.js LTS](https://nodejs.org/en)
* Run `npm ci`

## Building / Running / Testing stuff
This repo is managed by [Nx](https://nx.dev), a polyglot build system and
monorepo framework written in TypeScript. After installing dependencies, dev
tasks can be run through the Nx CLI:
```sh
npx nx <command> <project> [options]
# OR
npx nx run <project>:<command> [options]
```

For example, to compile the Bevy app to WebAssembly and serve it via the web
frontend project, run:
```
npx nx serve studio-frontend
```

You'll need to install the wasm compiler for Rust: 
```
rustup target add wasm32-unknown-unknown
```

Projects are located in directories at the workspace root. The list of commands
supported by a given project can be found under the `target` property of its
`project.json` file, which defines the Nx configuration for that project.

The full list of options for a given command can be discovered by passing the
`--help` flag:
```
npx nx <command> <project> --help
```
