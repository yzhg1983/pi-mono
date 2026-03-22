import type {
	ResponseFunctionWebSearch,
	ResponseOutputMessage,
	ResponseOutputText,
	ResponseReasoningItem,
	ResponseStreamEvent,
} from "openai/resources/responses/responses.js";
import { describe, expect, it } from "vitest";
import { convertResponsesMessages, processResponsesStream } from "../src/providers/openai-responses-shared.js";
import type { Api, AssistantMessage, AssistantMessageEvent, Model, TextContent } from "../src/types.js";
import { createAssistantMessageEventStream } from "../src/utils/event-stream.js";

function makeModel<TApi extends Api>(api: TApi): Model<TApi> {
	return {
		id: `model-${api}`,
		name: `Model ${api}`,
		api,
		provider: "openai",
		baseUrl: "https://api.openai.com/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 8192,
	};
}

function makeOutput(api: Api = "openai-responses"): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api,
		provider: "openai",
		model: `model-${api}`,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: 0,
	};
}

async function* makeStream(events: ResponseStreamEvent[]): AsyncIterable<ResponseStreamEvent> {
	for (const event of events) {
		yield event;
	}
}

async function collectEvents(
	stream: ReturnType<typeof createAssistantMessageEventStream>,
	finalMessage: AssistantMessage,
) {
	stream.end(finalMessage);
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) {
		events.push(event);
	}
	return events;
}

function makeReasoningItem(id = "rs_1"): ResponseReasoningItem {
	return {
		id,
		type: "reasoning",
		summary: [{ type: "summary_text", text: "Need to search first." }],
	};
}

function makeWebSearchItem(id: string, query: string): ResponseFunctionWebSearch {
	return {
		id,
		type: "web_search_call",
		status: "completed",
		action: {
			type: "search",
			query,
			queries: [query],
			sources: [{ type: "url", url: `https://example.com/${id}` }],
		},
	};
}

function makeOutputText(text: string): ResponseOutputText {
	return {
		type: "output_text",
		text,
		annotations: [
			{
				type: "url_citation",
				start_index: 0,
				end_index: text.length,
				title: "Example",
				url: "https://example.com/source",
			},
		],
	};
}

function makeMessageItem(id: string, text: string): ResponseOutputMessage {
	return {
		id,
		type: "message",
		role: "assistant",
		status: "completed",
		content: [makeOutputText(text)],
	};
}

describe("OpenAI Responses web search support", () => {
	it("stores completed web_search_call items on the assistant message", async () => {
		const output = makeOutput();
		const stream = createAssistantMessageEventStream();
		const completedSearch = makeWebSearchItem("ws_1", "latest pi-mono release");

		await processResponsesStream(
			makeStream([
				{
					type: "response.output_item.added",
					item: { id: "ws_1", type: "web_search_call", status: "in_progress" } as ResponseFunctionWebSearch,
					output_index: 0,
					sequence_number: 0,
				},
				{
					type: "response.output_item.done",
					item: completedSearch,
					output_index: 0,
					sequence_number: 1,
				},
			]),
			output,
			stream,
			makeModel("openai-responses"),
		);

		const events = await collectEvents(stream, output);

		expect(events).toEqual([
			{
				type: "server_toolcall_start",
				serverToolCall: { kind: "web_search", callId: "ws_1" },
				partial: output,
			},
			{
				type: "server_toolcall_end",
				serverToolCall: { kind: "web_search", callId: "ws_1" },
				partial: output,
			},
		]);
		expect(output.content).toHaveLength(0);
		expect(output._serverToolCalls).toEqual([JSON.stringify(completedSearch)]);
	});

	it("stores multiple web_search_call items in the original order", async () => {
		const output = makeOutput();
		const stream = createAssistantMessageEventStream();
		const firstSearch = makeWebSearchItem("ws_1", "first search");
		const secondSearch = makeWebSearchItem("ws_2", "second search");

		await processResponsesStream(
			makeStream([
				{
					type: "response.output_item.added",
					item: { id: "ws_1", type: "web_search_call", status: "in_progress" } as ResponseFunctionWebSearch,
					output_index: 0,
					sequence_number: 0,
				},
				{
					type: "response.output_item.done",
					item: firstSearch,
					output_index: 0,
					sequence_number: 1,
				},
				{
					type: "response.output_item.added",
					item: { id: "ws_2", type: "web_search_call", status: "in_progress" } as ResponseFunctionWebSearch,
					output_index: 1,
					sequence_number: 2,
				},
				{
					type: "response.output_item.done",
					item: secondSearch,
					output_index: 1,
					sequence_number: 3,
				},
			]),
			output,
			stream,
			makeModel("openai-responses"),
		);

		const events = await collectEvents(stream, output);

		expect(events).toEqual([
			{
				type: "server_toolcall_start",
				serverToolCall: { kind: "web_search", callId: "ws_1" },
				partial: output,
			},
			{
				type: "server_toolcall_end",
				serverToolCall: { kind: "web_search", callId: "ws_1" },
				partial: output,
			},
			{
				type: "server_toolcall_start",
				serverToolCall: { kind: "web_search", callId: "ws_2" },
				partial: output,
			},
			{
				type: "server_toolcall_end",
				serverToolCall: { kind: "web_search", callId: "ws_2" },
				partial: output,
			},
		]);
		expect(output._serverToolCalls).toEqual([JSON.stringify(firstSearch), JSON.stringify(secondSearch)]);
	});

	it("preserves output_text annotations on text content", async () => {
		const output = makeOutput();
		const stream = createAssistantMessageEventStream();

		await processResponsesStream(
			makeStream([
				{
					type: "response.output_item.added",
					item: {
						id: "msg_1",
						type: "message",
						role: "assistant",
						status: "in_progress",
						content: [],
					} as ResponseOutputMessage,
					output_index: 0,
					sequence_number: 0,
				},
				{
					type: "response.output_item.done",
					item: makeMessageItem("msg_1", "Current answer with citation"),
					output_index: 0,
					sequence_number: 1,
				},
			]),
			output,
			stream,
			makeModel("openai-responses"),
		);

		const textBlock = output.content[0] as TextContent;
		expect(textBlock.type).toBe("text");
		expect(textBlock.text).toBe("Current answer with citation");
		expect(textBlock.annotations).toEqual(makeOutputText("Current answer with citation").annotations);
	});

	it("replays stored web_search_call items after reasoning and before assistant messages in order", () => {
		const reasoning = makeReasoningItem();
		const firstSearch = makeWebSearchItem("ws_1", "first search");
		const secondSearch = makeWebSearchItem("ws_2", "second search");
		const model = makeModel("openai-responses");

		const messages = convertResponsesMessages(
			model,
			{
				messages: [
					{
						...makeOutput("openai-responses"),
						content: [
							{
								type: "thinking",
								thinking: "Need to search first.",
								thinkingSignature: JSON.stringify(reasoning),
							},
							{
								type: "text",
								text: "Answer with sources.",
								textSignature: JSON.stringify({ v: 1, id: "msg_1" }),
							},
						],
						_serverToolCalls: [JSON.stringify(firstSearch), JSON.stringify(secondSearch)],
					},
				],
			},
			new Set<string>(),
		);

		expect(messages).toHaveLength(4);
		expect((messages[0] as { type: string }).type).toBe("reasoning");
		expect((messages[1] as { type: string }).type).toBe("web_search_call");
		expect((messages[2] as { type: string }).type).toBe("web_search_call");
		expect((messages[3] as { type: string }).type).toBe("message");
		expect(messages[1]).toEqual(firstSearch);
		expect(messages[2]).toEqual(secondSearch);

		const replayedMessage = messages[3] as ResponseOutputMessage;
		expect(replayedMessage.content[0].type).toBe("output_text");
		if (replayedMessage.content[0].type === "output_text") {
			expect(replayedMessage.content[0].annotations).toEqual([]);
		}
	});

	it("replays stored web_search_call items across model switches within the same Responses API family", () => {
		const assistantMessage: AssistantMessage = {
			...makeOutput("openai-responses"),
			model: "gpt-4o",
			content: [{ type: "text", text: "Previous answer." }],
			_serverToolCalls: [JSON.stringify(makeWebSearchItem("ws_1", "search once"))],
		};
		const nextModel: Model<"openai-responses"> = {
			...makeModel("openai-responses"),
			id: "gpt-5",
			name: "GPT-5",
		};

		const messages = convertResponsesMessages(
			nextModel,
			{
				messages: [assistantMessage],
			},
			new Set<string>(),
		);

		expect(messages).toHaveLength(2);
		expect((messages[0] as { type: string }).type).toBe("web_search_call");
		expect((messages[1] as { type: string }).type).toBe("message");
	});

	it("skips replaying stored web_search_call items for different APIs", () => {
		const messages = convertResponsesMessages(
			makeModel("openai-completions"),
			{
				messages: [
					{
						...makeOutput("openai-responses"),
						content: [{ type: "text", text: "Previous answer." }],
						_serverToolCalls: [JSON.stringify(makeWebSearchItem("ws_1", "search once"))],
					},
				],
			},
			new Set<string>(),
		);

		expect(messages).toHaveLength(1);
		expect((messages[0] as { type: string }).type).toBe("message");
	});

	it("does not crash when a web_search_call completes with failed status", async () => {
		const output = makeOutput();
		const stream = createAssistantMessageEventStream();
		const failedSearch: ResponseFunctionWebSearch = {
			id: "ws_failed",
			type: "web_search_call",
			status: "failed",
			action: {
				type: "search",
				query: "failing search",
				queries: ["failing search"],
			},
		};

		await expect(
			processResponsesStream(
				makeStream([
					{
						type: "response.output_item.added",
						item: {
							id: "ws_failed",
							type: "web_search_call",
							status: "in_progress",
						} as ResponseFunctionWebSearch,
						output_index: 0,
						sequence_number: 0,
					},
					{
						type: "response.output_item.done",
						item: failedSearch,
						output_index: 0,
						sequence_number: 1,
					},
				]),
				output,
				stream,
				makeModel("openai-responses"),
			),
		).resolves.toBeUndefined();

		expect(output._serverToolCalls).toEqual([JSON.stringify(failedSearch)]);
	});

	it("keeps function calls working when web_search_call items are also present", async () => {
		const output = makeOutput();
		const stream = createAssistantMessageEventStream();
		const webSearch = makeWebSearchItem("ws_1", "latest express release");

		await processResponsesStream(
			makeStream([
				{
					type: "response.output_item.added",
					item: { id: "ws_1", type: "web_search_call", status: "in_progress" } as ResponseFunctionWebSearch,
					output_index: 0,
					sequence_number: 0,
				},
				{
					type: "response.output_item.done",
					item: webSearch,
					output_index: 0,
					sequence_number: 1,
				},
				{
					type: "response.output_item.added",
					item: {
						id: "fc_1",
						call_id: "call_1",
						type: "function_call",
						name: "write",
						arguments: '{"path":"package.json"}',
					},
					output_index: 1,
					sequence_number: 2,
				},
				{
					type: "response.function_call_arguments.done",
					item_id: "fc_1",
					output_index: 1,
					sequence_number: 3,
					name: "write",
					arguments: '{"path":"package.json"}',
				},
				{
					type: "response.output_item.done",
					item: {
						id: "fc_1",
						call_id: "call_1",
						type: "function_call",
						name: "write",
						arguments: '{"path":"package.json"}',
						status: "completed",
					},
					output_index: 1,
					sequence_number: 4,
				},
			]),
			output,
			stream,
			makeModel("openai-responses"),
		);

		const events = await collectEvents(stream, output);

		expect(events[0]).toEqual({
			type: "server_toolcall_start",
			serverToolCall: { kind: "web_search", callId: "ws_1" },
			partial: output,
		});
		expect(events[1]).toEqual({
			type: "server_toolcall_end",
			serverToolCall: { kind: "web_search", callId: "ws_1" },
			partial: output,
		});
		expect(events[2]?.type).toBe("toolcall_start");
		expect(events[3]?.type).toBe("toolcall_end");
		expect(output._serverToolCalls).toHaveLength(1);
		expect(output.content).toHaveLength(1);
		expect(output.content[0].type).toBe("toolCall");
		if (output.content[0].type === "toolCall") {
			expect(output.content[0].name).toBe("write");
			expect(output.content[0].arguments).toEqual({ path: "package.json" });
		}
	});
});
