#pragma once

// Unreal Core
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Engine/EngineTypes.h"

// EventCenterPlugin
#include "SpawnableActor.h"
#include "SpawnableStaticMeshActor.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(SpawnableStaticMeshActorLog, Log, All);

// NB(bt): "Blueprintable" seems to remove the ability to add BP event nodes.
UCLASS()
class ACTORSPAWNINGPLUGIN_API ASpawnableStaticMeshActor : public ASpawnableActor
{
	GENERATED_BODY()

public:

	ASpawnableStaticMeshActor();
	~ASpawnableStaticMeshActor();

	/** Access to the static mesh component */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SpawnableStaticMeshActor", meta = (AllowPrivateAccess = "true"))
	class UStaticMeshComponent* BaseMeshComponent;

	/** Set the static mesh. */
	UFUNCTION(BlueprintCallable, Category = "SpawnableStaticMeshActor")
	void SetSpawnableStaticMesh(UStaticMesh* staticMesh);

private:
	/** Mesh to use */
	//UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SpawnableActor", meta = (AllowPrivateAccess = "true"))
	class UStaticMesh* BaseMesh;
};

