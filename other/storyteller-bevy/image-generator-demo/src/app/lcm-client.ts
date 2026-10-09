export enum LcmLiveStatus {
	None = "NONE",
	Connected = "connected",
	Disconnected = "disconnected",
	Wait = "wait",
	SendFrame = "send_frame",
	Timeout = "timeout",
}

export interface ImageInfo {
	blob: Blob;
	prompt: string;
	negativePrompt: string;
	seed: number;
	guidanceScale: number;
}

interface LcmMessageData {
	status: LcmLiveStatus | "error";
	message?: string;
}

export class LcmClient extends WebSocket {
	static readonly Api = "//b3000.pancake-chimaera.ts.net/api";

	#userId: string;

	get status() { return this.#status; }
	#status = LcmLiveStatus.None;

	#waiting = false;

	#blob?: Blob;
	#prompt?: string;
	#negativePrompt?: string;
	#seed?: number;
	#guidanceScale = 0.8;

	constructor (userId: string) {
		super(`wss:${LcmClient.Api}/ws/${userId}`);
		this.#userId = userId;

		this.addEventListener("open", () => this.#onOpen());
		this.addEventListener("close", () => this.#onClose());
		this.addEventListener("message", event => this.#onMessage(event))
	}

	update(data: Partial<ImageInfo>): void {
		if ("blob" in data)
			this.#blob = data.blob;

		if ("prompt" in data)
			this.#prompt = data.prompt;

		if ("negativePrompt" in data)
			this.#negativePrompt = data.negativePrompt;

		if ("seed" in data)
			this.#seed = data.seed;

		this.#update();
	}

	#onOpen(): void {
		console.log(`Marking connection to websocket as open for user ${this.#userId}`);
		this.#status = LcmLiveStatus.Connected;
	}

	#onClose(): void {
		console.log(`Marking connection to websocket as closed for user ${this.#userId}`);
		this.#status = LcmLiveStatus.Disconnected;
	}

	#onMessage(event: MessageEvent<string>): void {
		console.log("onMessage:", event);
		const data: LcmMessageData = JSON.parse(event.data);

		if (data.status === "error") {
			this.#status = LcmLiveStatus.Disconnected;
			console.error(data.message);

			return;
		}

		this.#status = data.status;

		if (this.#status === LcmLiveStatus.SendFrame && this.#waiting)
			this.#sendFrame();
	}

	#update(): void {
		if (this.#status === LcmLiveStatus.SendFrame) {
			this.#sendFrame();
		} else {
			this.#waiting = true;
		}
	}

	async #sendFrame(): Promise<void> {
		if (!this.#blob || !this.#prompt || !this.#negativePrompt || this.#seed == null) {
			this.#waiting = true;
			return;
		}

		console.log("Sending next frame...");

		this.send(JSON.stringify({ status: "next_frame" }));
		this.send(JSON.stringify({
			prompt: this.#prompt,
			negativePrompt: this.#negativePrompt,
			seed: this.#seed,
			guidanceScale: this.#guidanceScale,
		}));
		this.send(this.#blob);

		this.#waiting = false;

		try {
			const url = `https:${LcmClient.Api}/stream/${this.#userId}`;

			// FIXME: This only actually renders an image in the DOM about ~10% of the time
			this.dispatchEvent(new CustomEvent("lcm-image-url-ready", { detail: url }));

			// const response = await fetch(url, { method: "GET" });
			// console.log("response:", response);

			// if (!response.ok || !response.body) {
			// 	console.error(`GET failed with status ${response.status}: "${response.statusText}"`);
			// 	return;
			// }

			// const blobParts = [] as Uint8Array[];
			// const reader = response.body.getReader();

			// // FIXME: This doesn't appear to be an image
			// // const { value } = await reader.read();
			// // if (value) blobParts.push(value);

			// // FIXME: this just stops responding after the first `read()` call even
			// //        though `done` is false
			// while (true) {
			// 	console.log("Reading next chunk...");
			//
			// 	const { value, done } = await reader.read();
			// 	console.log({ value, done });
			//
			// 	if (value) blobParts.push(value);
			// 	if (done) break;
			// }

			// this.dispatchEvent(new CustomEvent<Blob>("lcm-image-ready", {
			// 	detail: new Blob(blobParts, { type: "image/jpg" }),
			// }));
		}
		catch (err) {
			console.error(err);
		}
	}
}

export interface LcmClientEventMap extends WebSocketEventMap {
	"lcm-image-ready": CustomEvent<Blob>;
	"lcm-image-url-ready": CustomEvent<string>;
}

export interface LcmClient {
	addEventListener<K extends keyof LcmClientEventMap>(
		type: K,
		listener: (this: LcmClient, ev: LcmClientEventMap[K]) => any,
		options?: boolean | AddEventListenerOptions
	): void;
}
