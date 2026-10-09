#pragma once

// Unreal Core
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"

#include "ProtocolTypes.generated.h"

/**
 * Text to speech instructions.
 */
USTRUCT(BlueprintType)
struct EVENTCENTERPLUGIN_API FTtsRequest {
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString Username;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString VoiceName;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString Text;
};

/**
 * Spawn instructions
 */
USTRUCT(BlueprintType)
struct EVENTCENTERPLUGIN_API FSpawnRequest {
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString Username;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString Model;
};

/**
 * Warp instructions
 */
USTRUCT(BlueprintType)
struct EVENTCENTERPLUGIN_API FWarpRequest {
	GENERATED_BODY()

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString Username;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	FString LevelName;
};

