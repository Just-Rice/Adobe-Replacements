#pragma once

// Unreal Core
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"

// Function Library
#include "Kismet/BlueprintFunctionLibrary.h"

#include "ProtocolTypes.h"
#include "ProtocolLibrary.generated.h"

UCLASS()
class EVENTCENTERPLUGIN_API UProtocolLibrary : public UBlueprintFunctionLibrary 
{
	GENERATED_BODY()

public:
	// TODO: Take map as an input to make this even easier! Find the default key "default".

	// Decode TTS instructions
	UFUNCTION(BlueprintPure, Category = "EventCenterHelperFunctions")
	static FTtsRequest DecodeTtsPayload(const FString& payload);

	// Decode Spawn instructions
	UFUNCTION(BlueprintPure, Category = "EventCenterHelperFunctions")
	static FSpawnRequest DecodeSpawnPayload(const FString& payload);

	// Decode Warp instructions
	UFUNCTION(BlueprintPure, Category = "EventCenterHelperFunctions")
	static FWarpRequest DecodeWarpPayload(const FString& payload);
};
