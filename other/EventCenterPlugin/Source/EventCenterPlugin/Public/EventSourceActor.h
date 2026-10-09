#pragma once

// Unreal Core
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"

// Event system
#include "EventHandlerInterface.h"

// BlueprintWebSocketPlugin
// Docs: https://github.com/Pandoa/BlueprintWebSocket
#include "BlueprintWebSocketWrapper.h"

// EventCenterPlugin
#include "EventSourceActor.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(EventSourceLog, Log, All);

// TODO(bt): Make this not an actor. Consider UObject. Also consider a singleton.

// NB(bt): We can't use `DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams` multicast 
// due to an Unreal Engine limitation
DECLARE_DYNAMIC_DELEGATE_TwoParams(FMySubscriberRefDelegate, const FString&, Channel, const FString&, Message);


/**
 * AEventSourceActor
 *
 * Manages an underlying Redis pubsub subscription and coordinates dispatch to other actors.
 */

// NB(bt): "Blueprintable" seems to remove the ability to add BP event nodes.
UCLASS()
class EVENTCENTERPLUGIN_API AEventSourceActor : public AActor
{
	GENERATED_BODY()

public:

	AEventSourceActor();
	~AEventSourceActor();

	/** Set up websockets.*/
	UFUNCTION(BlueprintCallable, Category = "EventCenter")
	void SetupWebSockets();

	/** 
	 * Bind a delegate to a channel. 
	 * 
	 * Note that we can't multicast due to Unreal limitations. As a workaround, bind 
	 * different channels for other events,or have the events delegates themselves 
	 * perform the multicasting downstream of us.
	 */
	UFUNCTION(BlueprintCallable, Category = "EventCenter")
	void BindSingleDelegateForChannel(const FString& channel, 
									  const FMySubscriberRefDelegate& delegateDef);

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	virtual void BeginDestroy() override;

	// websocket callbacks
	UFUNCTION() void OnConnected();
	UFUNCTION() void OnConnectionError(const FString& Error);
	UFUNCTION() void OnClosed(int64 StatusCode, const FString& Reason, bool bWasClean);
	UFUNCTION() void OnMessage(const FString& Message);
	UFUNCTION() void OnRawMessage(const TArray<uint8>& Data, int32 BytesRemaining);
	UFUNCTION() void OnMessageSent(const FString& Message);

private:
	UPROPERTY()
	UBlueprintWebSocket* WebSocket;

	// TODO(bt): Value type can be wrapped in a class to add additional routing. 
	// Multiple delegation signatures, "pause/muting", etc.
    TMap<FString, FMySubscriberRefDelegate> delegateSubscriptions;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
};
