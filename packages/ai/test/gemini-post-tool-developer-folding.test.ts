import { describe, expect, it } from "bun:test";
import { convertMessages as convertGoogleMessages } from "@oh-my-pi/pi-ai/providers/google-shared";
import { convertMessages as convertChatCompletionsMessages } from "@oh-my-pi/pi-ai/providers/openai-completions";
import { buildResponsesInput } from "@oh-my-pi/pi-ai/providers/openai-shared";
import type {
	AssistantMessage,
	Context,
	DeveloperMessage,
	ImageContent,
	Model,
	ModelSpec,
	ResolvedOpenAICompat,
	ToolResultMessage,
	Usage,
	UserMessage,
} from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";

const ZERO_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const SAMPLE_IMAGE: ImageContent = {
	type: "image",
	mimeType: "image/png",
	data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
};

function makeOpenRouterGeminiModel(): Model<"openrouter"> {
	return buildModel({
		id: "google/gemini-4-neon",
		name: "Gemini 4 Neon",
		api: "openrouter",
		provider: "openrouter",
		baseUrl: "https://openrouter.ai/api/v1",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_048_576,
		maxTokens: 262_144,
	} as ModelSpec<"openrouter">);
}

function makeOpenRouterClaudeModel(): Model<"openrouter"> {
	return buildModel({
		id: "anthropic/claude-sonnet-4-6",
		name: "Claude Sonnet 4.6",
		api: "openrouter",
		provider: "openrouter",
		baseUrl: "https://openrouter.ai/api/v1",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200_000,
		maxTokens: 64_000,
	} as ModelSpec<"openrouter">);
}

function makeVertexGeminiModel(): Model<"google-vertex"> {
	return buildModel({
		id: "gemini-4-neon",
		name: "Gemini 4 Neon",
		api: "google-vertex",
		provider: "google-vertex",
		baseUrl: "https://aiplatform.googleapis.com",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_048_576,
		maxTokens: 262_144,
	});
}

function makeToolTurn(opts?: {
	provider?: string;
	api?: "openrouter" | "google-vertex";
	model?: string;
}): [UserMessage, AssistantMessage, ToolResultMessage] {
	const provider = opts?.provider ?? "openrouter";
	const api = opts?.api ?? "openrouter";
	const model = opts?.model ?? "google/gemini-4-neon";
	return [
		{
			role: "user",
			content: "Inspect workspace-page-loader.ts",
			attribution: "user",
			timestamp: 1000,
		},
		{
			role: "assistant",
			provider,
			api,
			model,
			content: [
				{
					type: "toolCall",
					id: "call_read_1",
					name: "read",
					arguments: { path: "workspace-page-loader.ts" },
				},
			],
			usage: ZERO_USAGE,
			stopReason: "toolUse",
			timestamp: 2000,
		},
		{
			role: "toolResult",
			toolCallId: "call_read_1",
			toolName: "read",
			content: [{ type: "text", text: "export function loadWorkspace() {}" }],
			isError: false,
			timestamp: 3000,
		},
	];
}

describe("Gemini post-toolResult developer notice folding", () => {
	it("folds consecutive agent-attributed developer notices into function_call_output on OpenRouter Responses", () => {
		const model = makeOpenRouterGeminiModel();
		const [user, assistant, toolResult] = makeToolTurn();
		const todoNudge: DeveloperMessage = {
			role: "developer",
			content: [{ type: "text", text: "<system-reminder>\n3 todo items still open.\n</system-reminder>" }],
			attribution: "agent",
			timestamp: 3100,
		};
		const asyncResult: DeveloperMessage = {
			role: "developer",
			content: [{ type: "text", text: "<system-notice>\nBackground job bg_1 completed.\n</system-notice>" }],
			attribution: "agent",
			timestamp: 3200,
		};

		const input = buildResponsesInput({
			model,
			context: { messages: [user, assistant, toolResult, todoNudge, asyncResult] },
			strictResponsesPairing: false,
			supportsImageDetailOriginal: false,
		});

		const outputItems = input.filter(item => item.type === "function_call_output");
		expect(outputItems).toHaveLength(1);
		expect(input.at(-1)?.type).toBe("function_call_output");
		expect(outputItems[0]).toMatchObject({
			type: "function_call_output",
			call_id: "call_read_1",
			output:
				"export function loadWorkspace() {}\n\n" +
				"<system-reminder>\n3 todo items still open.\n</system-reminder>\n\n" +
				"<system-notice>\nBackground job bg_1 completed.\n</system-notice>",
		});
	});

	it("folds agent-attributed developer notices into role: tool on OpenRouter Chat Completions", () => {
		const model = makeOpenRouterGeminiModel();
		const [user, assistant, toolResult] = makeToolTurn();
		const todoNudge: DeveloperMessage = {
			role: "developer",
			content: "<system-reminder>\nUpdate todo status.\n</system-reminder>",
			attribution: "agent",
			timestamp: 3100,
		};

		const messages = convertChatCompletionsMessages(
			model as unknown as Model<"openai-completions">,
			{ messages: [user, assistant, toolResult, todoNudge] },
			model.compat as ResolvedOpenAICompat,
		);

		expect(messages.at(-1)).toEqual({
			role: "tool",
			tool_call_id: "call_read_1",
			content: "export function loadWorkspace() {}\n\n<system-reminder>\nUpdate todo status.\n</system-reminder>",
		});
	});

	it("folds agent-attributed developer notices into functionResponse.response.output on Vertex Gemini", () => {
		const model = makeVertexGeminiModel();
		const [user, assistant, toolResult] = makeToolTurn({
			provider: "google-vertex",
			api: "google-vertex",
			model: "gemini-4-neon",
		});
		const todoNudge: DeveloperMessage = {
			role: "developer",
			content: [{ type: "text", text: "<system-reminder>\nKeep going.\n</system-reminder>" }],
			attribution: "agent",
			timestamp: 3100,
		};
		const context: Context = { messages: [user, assistant, toolResult, todoNudge] };

		const contents = convertGoogleMessages(model, context);

		expect(contents).toHaveLength(3);
		expect(contents.at(-1)).toEqual({
			role: "user",
			parts: [
				{
					functionResponse: {
						name: "read",
						response: {
							output: "export function loadWorkspace() {}\n\n<system-reminder>\nKeep going.\n</system-reminder>",
						},
					},
				},
			],
		});
	});

	it("preserves user-attributed developer turns, image-bearing developer turns, and non-Gemini models as separate messages", () => {
		const geminiModel = makeOpenRouterGeminiModel();
		const claudeModel = makeOpenRouterClaudeModel();
		const [user, assistant, toolResult] = makeToolTurn();

		const userFileMention: DeveloperMessage = {
			role: "developer",
			content: [{ type: "text", text: '<file path="notes.md">\nspec\n</file>' }],
			attribution: "user",
			timestamp: 3100,
		};
		const withUserMention = convertChatCompletionsMessages(
			geminiModel as unknown as Model<"openai-completions">,
			{ messages: [user, assistant, toolResult, userFileMention] },
			geminiModel.compat as ResolvedOpenAICompat,
		);
		expect(withUserMention.slice(-2).map(m => m.role)).toEqual(["tool", "user"]);

		const imageDeveloper: DeveloperMessage = {
			role: "developer",
			content: [{ type: "text", text: "Captured frame" }, SAMPLE_IMAGE],
			attribution: "agent",
			timestamp: 3100,
		};
		const withImage = convertChatCompletionsMessages(
			geminiModel as unknown as Model<"openai-completions">,
			{ messages: [user, assistant, toolResult, imageDeveloper] },
			geminiModel.compat as ResolvedOpenAICompat,
		);
		expect(withImage.slice(-2).map(m => m.role)).toEqual(["tool", "user"]);

		const agentNudge: DeveloperMessage = {
			role: "developer",
			content: [{ type: "text", text: "<system-reminder>\n3 todo items still open.\n</system-reminder>" }],
			attribution: "agent",
			timestamp: 3100,
		};
		const onClaude = convertChatCompletionsMessages(
			claudeModel as unknown as Model<"openai-completions">,
			{ messages: [user, assistant, toolResult, agentNudge] },
			claudeModel.compat as ResolvedOpenAICompat,
		);
		expect(onClaude.slice(-2).map(m => m.role)).toEqual(["tool", "user"]);
	});
});
