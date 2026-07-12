"use strict";

const { execFileSync } = require("node:child_process");
const pins = require("../pins.cjs");

const actualNode = process.version;
const expectedNode = `v${pins.node}`;
if (actualNode !== expectedNode) {
  throw new Error(`expected Node.js ${expectedNode}, got ${actualNode}`);
}

const actualNpm = execFileSync("npm", ["--version"], { encoding: "utf8" }).trim();
if (actualNpm !== pins.npm) {
  throw new Error(`expected npm ${pins.npm}, got ${actualNpm}`);
}

console.log(`exact JavaScript runtime selected: Node.js ${actualNode}, npm ${actualNpm}`);
