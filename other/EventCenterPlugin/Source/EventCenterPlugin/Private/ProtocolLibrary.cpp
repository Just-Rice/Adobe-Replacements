#include "ProtocolLibrary.h"

FTtsRequest UProtocolLibrary::DecodeTtsPayload(const FString& payload)
{
	FString input = payload;

	FString username = TEXT("");
	FString voiceName = TEXT("");
	FString text = TEXT("");

	FString Left, Right;

	if (input.Split(TEXT("|"), &Left, &input)) {
		username = Left;

		if (input.Split(TEXT(" "), &Left, &Right)) {
			voiceName = Left;
			text = Right;
		}
	}

	FTtsRequest request = FTtsRequest();
	request.Username = username.TrimStartAndEnd();
	request.VoiceName = voiceName.TrimStartAndEnd();
	request.Text = text.TrimStartAndEnd();

	return request;
}

FSpawnRequest UProtocolLibrary::DecodeSpawnPayload(const FString& payload)
{
	FString input = payload;

	FString username = TEXT("");
	FString model = TEXT("");

	FString Left, Right;

	if (input.Split(TEXT("|"), &Left, &Right)) {
		username = Left;
		model = Right;
	}

	FSpawnRequest request = FSpawnRequest();
	request.Username = username.TrimStartAndEnd();
	request.Model = model.TrimStartAndEnd();

	return request;
}

FWarpRequest UProtocolLibrary::DecodeWarpPayload(const FString& payload)
{
	FString input = payload;

	FString username = TEXT("");
	FString level = TEXT("");

	FString Left, Right;

	if (input.Split(TEXT("|"), &Left, &Right)) {
		username = Left;
		level = Right;
	}

	FWarpRequest request = FWarpRequest();
	request.Username = username.TrimStartAndEnd();
	request.LevelName = level.TrimStartAndEnd();

	return request;
}
