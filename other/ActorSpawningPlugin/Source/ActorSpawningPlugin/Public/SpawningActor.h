#pragma once

// Unreal Core
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"

// EventCenterPlugin
#include "SoundSpawnActor.h"
#include "SpawnableActor.h"
#include "SpawnableSkeletalMeshActor.h"
#include "SpawnableStaticMeshActor.h"
#include "SpawningActor.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(SpawningActorLog, Log, All);

// NB(bt): "Blueprintable" seems to remove the ability to add BP event nodes.
UCLASS()
class ACTORSPAWNINGPLUGIN_API ASpawningActor : public AActor
{
	GENERATED_BODY()

public:

	ASpawningActor();
	~ASpawningActor();

	// ========================== STATIC MESH ACTOR ========================== 

	/** 
	 * Create static mesh actor.
	 */
	UFUNCTION(BlueprintCallable, Category = "SpawningActor")
	ASpawnableStaticMeshActor* SpawnStaticMeshActor();

	// ========================== SKELETAL MESH ACTOR ========================== 

	/**
	 * Create skeletal mesh actor.
	 */
	UFUNCTION(BlueprintCallable, Category = "SpawningActor")
	ASpawnableSkeletalMeshActor* SpawnSkeletalMeshActor();

	/**
	 * Create skeletal mesh actor at location.
	 */
	UFUNCTION(BlueprintCallable, Category = "SpawningActor")
	ASpawnableSkeletalMeshActor* SpawnSkeletalMeshActorAtLocation(FVector location);

	// ========================== SOUND ACTOR ========================== 

	/** 
	 * Create sound actor.
	 */
	UFUNCTION(BlueprintCallable, Category = "SpawningActor")
	ASoundSpawnActor* SpawnSoundActor();

	/** 
	 * Create sound actor at the given location.
	 */
	UFUNCTION(BlueprintCallable, Category = "SpawningActor")
	ASoundSpawnActor* SpawnSoundActorAtLocation(FVector location);

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	virtual void BeginDestroy() override;

	// Lookup the player's location.
	FVector LookupPlayerLocation();
};
