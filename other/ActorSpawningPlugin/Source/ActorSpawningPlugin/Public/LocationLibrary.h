#pragma once

// Unreal Core
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"

// Function Library
#include "Kismet/BlueprintFunctionLibrary.h"

#include "LocationLibrary.generated.h"

/**
 * Helper functions for obtaining and manipulating locations.
 * The world context is automatically injected.
 */
UCLASS()
class ACTORSPAWNINGPLUGIN_API ULocationLibrary : public UBlueprintFunctionLibrary 
{
	GENERATED_BODY()

public:
	// ========================== ABSOLUTE LOCATION ========================== 

	/**
	 * The player's location
	 * This has no relation to where the player is looking, but rather the global map.
	 */
	UFUNCTION(BlueprintPure, Category = "SpawningPluginLocationFunctions", 
		meta = (WorldContext = "WorldContextObject", UnsafeDuringActorConstruction = "true"))
	static FVector GetPlayerAbsoluteLocation(UObject* WorldContextObject);

	/**
	 * The player's location with a static offset
	 * This has no relation to where the player is looking, but rather the global map.
	 */
	UFUNCTION(BlueprintPure, Category = "SpawningPluginLocationFunctions",
		meta = (WorldContext = "WorldContextObject", UnsafeDuringActorConstruction = "true"))
	static FVector GetPlayerAbsoluteLocationWithOffset(UObject* WorldContextObject, 
		float x, float y, float z);

	/**
	 * The player's location with a random offset
	 * This has no relation to where the player is looking, but rather the global map.
	 */
	UFUNCTION(BlueprintPure, Category = "SpawningPluginLocationFunctions",
		meta = (WorldContext = "WorldContextObject", UnsafeDuringActorConstruction = "true"))
	static FVector GetPlayerAbsoluteLocationWithRandomOffset(UObject* WorldContextObject,
		float minX, float maxX, float minY, 
		float maxY, float minZ, float maxZ);

	// ========================== FORWARD VECTOR ========================== 

	/**
	 * The player's forward location with a static offset
	 * This appears "in front" of the player.
	 * The xyz-offsets are applied before the forward offset.
	 */
	UFUNCTION(BlueprintPure, Category = "SpawningPluginLocationFunctions",
		meta = (WorldContext = "WorldContextObject", UnsafeDuringActorConstruction = "true"))
	static FVector GetPlayerForwardLocationWithOffset(UObject* WorldContextObject, 
		float forwardOffset,
		float xAbsoluteOffset, float yAbsoluteOffset, float zAbsoluteOffset);

	/**
	 * The player's forward location with a random offset
	 * This appears "in front" of the player.
	 * The xyz-offsets are applied before the forward offset.
	 */
	UFUNCTION(BlueprintPure, Category = "SpawningPluginLocationFunctions",
		meta = (WorldContext = "WorldContextObject", UnsafeDuringActorConstruction = "true"))
	static FVector GetPlayerForwardLocationWithRandomOffset(UObject* WorldContextObject,
		float minForwardOffset, float maxForwardOffset,
		float minAbsoluteXOffset, float maxAbsoluteXOffset, 
		float minAbsoluteYOffset, float maxAbsoluteYOffset, 
		float minAbsoluteZOffset, float maxAbsoluteZOffset);


private:
	static APawn* GetPlayerPawn(UObject* WorldContextObject);
};
