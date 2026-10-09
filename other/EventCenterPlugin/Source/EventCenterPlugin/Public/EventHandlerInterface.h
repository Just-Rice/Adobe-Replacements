#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "UObject/Interface.h"

#include "EventHandlerInterface.generated.h"

/**
 * Interface for all objects that can handle events. 
 * 
 * My C++ foo isn't strong enough to accomplish this with templates, lambdas, 
 * or reuse of unreal's existing delegation system.
 *
 * Assuming this works, see:
 * https://docs.unrealengine.com/en-US/ProgrammingAndScripting/GameplayArchitecture/Interfaces/index.html
 */
UINTERFACE(BlueprintType)
class EVENTCENTERPLUGIN_API UEventHandlerInterface : public UInterface
{
    GENERATED_BODY()

/*public:
    UEventHandlerInterface() {}
    virtual ~UEventHandlerInterface() {}*/

};


class EVENTCENTERPLUGIN_API IEventHandlerInterface
{
    GENERATED_BODY()

public:
    // Virtual functions cannot have UFUNCTION specifiers.
    //virtual void HandleEvent(FString channel, FString message) = 0;

    // ^ NB(bt): Designation of "= 0" makes this a pure virtual method,
    // and it also makes the class abstract.

    // Event callback for events dispatched with EventCenterPlugin.
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Event Callback")
    void HandleEvent2(const FString& channel, const FString& message);
};
