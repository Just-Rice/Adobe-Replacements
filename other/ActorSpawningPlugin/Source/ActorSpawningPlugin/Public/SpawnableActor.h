#pragma once

// Unreal Core
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Engine/EngineTypes.h"

// EventCenterPlugin
#include "SpawnableActor.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(SpawnableActorLog, Log, All);

// NB(bt): "Blueprintable" seems to remove the ability to add BP event nodes.
UCLASS()
class ACTORSPAWNINGPLUGIN_API ASpawnableActor : public AActor
{
	GENERATED_BODY()

public:

	ASpawnableActor();
	~ASpawnableActor();

	/** Set a timer to destroy this actor. */
	void StartDestroyTimer(float seconds);

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	virtual void BeginDestroy() override;

	// Destroy timer callback.
	void InvokeDestroy();

	// (Optional) destroy callback timer.
	FTimerHandle DestroyTimer;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
};

