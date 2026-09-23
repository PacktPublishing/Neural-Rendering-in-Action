
#import "PlasmaView.h"

#include "pch.h"
#include "Plasma.h"
extern Plasma* gpApp;

@implementation PlasmaView {
}

- (instancetype)initWithCoder:(NSCoder *)coder {
    self = [super initWithCoder:coder];
    if(self) {
        [self registerForDraggedTypes:@[NSPasteboardTypeFileURL]];
    }
    return self;
}

- (void)viewDidMoveToWindow {
    [self.window setAcceptsMouseMovedEvents:YES];
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)keyDown:(NSEvent *)event {
}

- (void)keyUp:(NSEvent *) event {
    gpApp->onKeyPressed(event.charactersIgnoringModifiers, event.modifierFlags);
}

- (void)mouseMoved:(NSEvent *)event {
    NSPoint location = [event locationInWindow];
    gpApp->onMouseMoved(location.x, location.y, false, false, false);
}

- (void)mouseDown:(NSEvent *)event {
}

- (void)mouseUp:(NSEvent *)event {
}

- (void)mouseDragged:(NSEvent *)event {
    NSPoint location = [event locationInWindow];
    gpApp->onMouseMoved(location.x, location.y, true, false, false);
}

- (void)rightMouseDown:(NSEvent *)event {
}

- (void)rightMouseUp:(NSEvent *)event {
}

- (void)rightMouseDragged:(NSEvent *)event {
    NSPoint location = [event locationInWindow];
    gpApp->onMouseMoved(location.x, location.y, false, false, true);
}

- (void)otherMouseDown:(NSEvent *)event {
}

- (void)otherMouseUp:(NSEvent *)event {
}

- (void)otherMouseDragged:(NSEvent *)event {
    NSPoint location = [event locationInWindow];
    gpApp->onMouseMoved(location.x, location.y, false, true, false);
}

- (void)scrollWheel:(NSEvent *)event {
    CGFloat deltaY = event.scrollingDeltaY;
    gpApp->onMouseWheel(deltaY > 0.0f ? 1 : -1);
}

- (NSDragOperation)draggingEntered:(id<NSDraggingInfo>)sender {
    if([[sender draggingPasteboard].types containsObject:NSPasteboardTypeFileURL]) {
        return NSDragOperationCopy;
    }
    return NSDragOperationNone;
}

- (void)draggingExited:(nullable id<NSDraggingInfo>)sender {
}

- (BOOL)performDragOperation:(id<NSDraggingInfo>)sender {
    NSPasteboard* pasteboard = [sender draggingPasteboard];
    NSArray<NSPasteboardItem*>* items = [pasteboard pasteboardItems];
    BOOL acceptedData = NO;
    for(NSPasteboardItem* item in items) {
        NSString* fileURLString = [item stringForType:NSPasteboardTypeFileURL] ? [item stringForType:NSPasteboardTypeFileURL] : [item stringForType:NSPasteboardTypeURL];
        if(fileURLString) {
            NSURL* fileURL = [NSURL URLWithString:fileURLString];
            if(fileURL && fileURL.isFileURL) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    gpApp->onFilesDropped(fileURL);
                });
                acceptedData = YES;
            }
        }
    }
    return acceptedData;
}

- (void)concludeDragOperation:(nullable id<NSDraggingInfo>)sender {
}

- (void)draggingEnded:(id<NSDraggingInfo>) sender {
}

@end
