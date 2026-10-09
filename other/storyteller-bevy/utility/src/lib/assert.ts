export function assert(expression: unknown, message?: string): asserts expression {
	if (!expression) {
		let err = "Assertion failed";
		if (message) err += `: ${message}`;

		throw new Error(err);
	}
}
