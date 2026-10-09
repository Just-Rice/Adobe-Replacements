#include "EventSourceActor.h"

#include "UObject/Object.h"

DEFINE_LOG_CATEGORY(EventSourceLog);

AEventSourceActor::AEventSourceActor() :
	WebSocket(nullptr)
{
	// Set this actor to call Tick() every frame. 
	// You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;
}

AEventSourceActor::~AEventSourceActor()
{
}

void AEventSourceActor::BeginPlay()
{
	Super::BeginPlay();
}

void AEventSourceActor::BeginDestroy()
{
	Super::BeginDestroy();
}

void AEventSourceActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	// TODO(bt): If our callbacks aren't enough to restore broken connections, add some logic here.
}

// ======================================= web sockets =============================================

void AEventSourceActor::OnConnected() 
{
	GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "OnConnected");
	UE_LOG(EventSourceLog, Log, TEXT("OnConnected"));
}

void AEventSourceActor::OnConnectionError(const FString& Error)
{
	GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "OnConnectionError");
	UE_LOG(EventSourceLog, Error, TEXT("OnConnectionError. Failed to connect: %s."), *Error);
}

void AEventSourceActor::OnClosed(int64 StatusCode, const FString& Reason, bool bWasClean)
{
	GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "OnClosed");
	UE_LOG(EventSourceLog, Warning, TEXT("OnClosed. Connection closed: %d:%s. Clean: %d"), StatusCode, *Reason, bWasClean);
}

void AEventSourceActor::OnMessage(const FString& Message)
{
	GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "OnMessage");
	UE_LOG(EventSourceLog, Log, TEXT("OnMessage. New message: %s"), *Message);

	FString channel, payload;
	bool result = Message.Split(TEXT("|"), &channel, &payload);

	if (!result) {
		UE_LOG(EventSourceLog, Log, TEXT("Could not split"));
		GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "Could not split");
		return;
	}

	if (!delegateSubscriptions.Contains(channel)) {
		UE_LOG(EventSourceLog, Log, TEXT("No delegate"));
		GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "No delegate");
		return;
	}

	UE_LOG(EventSourceLog, Log, TEXT("Executing delegate"));
	GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "Executing delegate");
	delegateSubscriptions[channel].Execute(channel, payload);

}

void AEventSourceActor::OnRawMessage(const TArray<uint8>& Data, int32 BytesRemaining)
{
	GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "OnRawMessage");
	UE_LOG(EventSourceLog, Log, TEXT("OnRawMessage. New binary message: %d bytes and %d bytes remaining."), Data.Num(), BytesRemaining);
}

void AEventSourceActor::OnMessageSent(const FString& Message)
{
	GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "OnMessageSent");
	UE_LOG(EventSourceLog, Log, TEXT("OnMessageSent. We just sent %s to the server."), *Message);
}

void AEventSourceActor::SetupWebSockets()
{
	GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, "SetupWebSockets");
	UE_LOG(EventSourceLog, Log, TEXT("SetupWebSockets"));

	if (WebSocket == nullptr) {
		WebSocket = UBlueprintWebSocket::CreateWebSocket();
	}

	WebSocket->OnConnectedEvent.AddDynamic(this, &AEventSourceActor::OnConnected);
	WebSocket->OnConnectionErrorEvent.AddDynamic(this, &AEventSourceActor::OnConnectionError);
	WebSocket->OnCloseEvent.AddDynamic(this, &AEventSourceActor::OnClosed);
	WebSocket->OnMessageEvent.AddDynamic(this, &AEventSourceActor::OnMessage);
	WebSocket->OnRawMessageEvent.AddDynamic(this, &AEventSourceActor::OnRawMessage);
	WebSocket->OnMessageSentEvent.AddDynamic(this, &AEventSourceActor::OnMessageSent);

	WebSocket->Connect(TEXT("ws://localhost:8765/"), TEXT("ws"));
}

void AEventSourceActor::BindSingleDelegateForChannel(const FString& channel, const FMySubscriberRefDelegate& delegateDef)
{
	delegateSubscriptions.Add(channel, delegateDef);
}
