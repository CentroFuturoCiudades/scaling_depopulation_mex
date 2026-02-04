function RegisterInteractiveHandlers() {
RegisterButtonEventHandlers();
ProcessAnimations();
ProcessMedia();
}
function ProcessMedia() {
var oFrame = document.getElementsByClassName("_idGenMedia");
for (var i = 0; i < oFrame.length; i++) {
var actions = oFrame[i].getAttribute("data-mediaOnPageLoadActions");
if(actions) {
var descendants = oFrame[i].getElementsByTagName('*');
for(var j = 0; j < descendants.length; j++) {
var e = descendants[j];
var tagName = e.tagName.toLowerCase();
if(tagName == 'video' || tagName == 'audio') {
if(e.paused) {
var selfContainerID = e.id;
eval(actions);
}
}
}
}
}
}
function ProcessAnimations() {
	var oFrame = document.getElementsByClassName("_idGenAnimation");
	for (var i = 0; i < oFrame.length; i++) {
		var actions = oFrame[i].getAttribute("data-animationOnPageLoadActions");
		if(actions) {
			var selfContainerID = oFrame[i].id
			eval(actions);
		}
		var cn = oFrame[i].className;
		if(cn.indexOf("_idGenCurrentState") != -1) {
			var actions = oFrame[i].getAttribute("data-animationOnStateLoadActions");
			if(actions) {
				var selfContainerID = oFrame[i].id
				eval(actions);
			}
		}
		actions = oFrame[i].getAttribute("data-animationOnSelfClickActions");
		if(actions) {
			oFrame[i].addEventListener("touchend", function(event) { onTouchEndForAnimations(this, event) }, false);
			oFrame[i].addEventListener("mouseup", function(event) { onMouseUpForAnimations(this, event) }, false);
		}
		actions = oFrame[i].getAttribute("data-animationOnSelfRolloverActions");
		if(actions) {
			oFrame[i].addEventListener("mouseover", function(event) { onMouseOverForAnimations(this, event) }, false);
		}
	}
	document.body.addEventListener("touchend", function(event) { onPageTouchEndForAnimations(this, event) }, false);
	document.body.addEventListener("mouseup", function(event) { onPageMouseUpForAnimations(this, event) }, false);
}
function onPageTouchEndForAnimations(element, event) {
var oFrame = document.getElementsByClassName("_idGenAnimation");
for (var i = 0; i < oFrame.length; i++) {
var actions = oFrame[i].getAttribute("data-animationOnPageClickActions");
if(actions) {
var selfContainerID = oFrame[i].id;
eval(actions);
event.target.__id_touched = true;
}
}
event.stopPropagation();
}
function onPageMouseUpForAnimations(element, event) {
if (event.target && event.target.__id_touched) {event.target.__id_touched=false; return;}
var oFrame = document.getElementsByClassName("_idGenAnimation");
for (var i = 0; i < oFrame.length; i++) {
var actions = oFrame[i].getAttribute("data-animationOnPageClickActions");
if(actions) {
var selfContainerID = oFrame[i].id;
eval(actions);
}
}
event.stopPropagation();
}
function onTouchEndForAnimations(element, event) {
var classID = element.getAttribute("data-animationObjectType");
var oFrame = document.getElementsByClassName(classID);
for (var i = 0; i < oFrame.length; i++) {
var actions = oFrame[i].getAttribute("data-animationOnSelfClickActions");
if(actions) {
var selfContainerID = oFrame[i].id;
eval(actions);
event.target.__id_touched = true;
}
}
event.stopPropagation();
event.preventDefault();
}
function onMouseUpForAnimations(element, event) {
if (event.target && event.target.__id_touched) {event.target.__id_touched=false; return;}
var classID = element.getAttribute("data-animationObjectType");
var oFrame = document.getElementsByClassName(classID);
for (var i = 0; i < oFrame.length; i++) {
var actions = oFrame[i].getAttribute("data-animationOnSelfClickActions");
if(actions) {
var selfContainerID = oFrame[i].id;
eval(actions);
}
}
event.stopPropagation();
event.preventDefault();
}
function onMouseOverForAnimations(element, event) {
var animationClassName = element.getAttribute("data-idGenAnimationClass");
var cn = element.className;
if (cn.indexOf(animationClassName) != -1 ) {
return;
}
var classID = element.getAttribute("data-animationObjectType");
var oFrame = document.getElementsByClassName(classID);
for (var i = 0; i < oFrame.length; i++) {
var actions = oFrame[i].getAttribute("data-animationOnSelfRolloverActions");
if(actions) {
var selfContainerID = oFrame[i].id;
eval(actions);
}
}
event.stopPropagation();
event.preventDefault();
}
function RegisterButtonEventHandlers() {
var oFrame = document.getElementsByClassName("_idGenButton");
for (var i = 0; i < oFrame.length; i++) {
oFrame[i].addEventListener("touchstart", function(event) { onTouchStart(this, event) }, false);
oFrame[i].addEventListener("touchend", function(event) { onTouchEnd(this, event) }, false);
oFrame[i].addEventListener("mousedown", function(event) { onMouseDown(this, event) }, false);
oFrame[i].addEventListener("mouseup", function(event) { onMouseUp(this, event) }, false);
oFrame[i].addEventListener("mouseover", function(event) { onMouseOver(this, event) }, false);
oFrame[i].addEventListener("mouseout", function(event) { onMouseOut(this, event) }, false);
}
}
function hasAppearance(element, appearance) {
var childArray = element.children;
for(var i=0; i< childArray.length; i++) {
var cn = childArray[i].className;
if(cn.indexOf(appearance) != -1) {
return true;
}
}
return false;
}
function isDescendantOf(child, parent) {
var current = child;
while(current) {
if(current == parent)
return true;
current = current.parentNode;
}
return false;
}
function addClass(element,classname) { 
var cn = element.className;
if (cn.indexOf(classname) != -1 ) {
return;
}
if (cn != '') {
classname = ' ' + classname;
}
element.className = cn + classname;
}
function removeClass(element, classname) {
var cn = element.className;
var rxp = new RegExp("\\s?\\b" + classname + "\\b", "g");
cn = cn.replace(rxp, '');
element.className = cn;
}
function onMouseDown(element, event) {
if (event.target && event.target.__id_touched) {event.target.__id_touched=false; return;}
if (hasAppearance(element, '_idGen-Appearance-Click')) {
addClass(element, '_idGenStateClick');
}
var actions = element.getAttribute("data-clickactions");
if(actions) {
eval(actions);
}
event.stopPropagation();
event.preventDefault();
}
function onMouseUp(element, event) {
if (event.target && event.target.__id_touched) {event.target.__id_touched=false; return;}
removeClass(element, '_idGenStateClick');
var actions = element.getAttribute("data-releaseactions");
if(actions) {
eval(actions);
}
event.stopPropagation();
event.preventDefault();
}
function onMouseOver(element, event) {
if (event.relatedTarget) {
if(isDescendantOf(event.relatedTarget, element)) return;
}
if (hasAppearance(element, '_idGen-Appearance-Rollover')) {
addClass(element, '_idGenStateHover');
}
var actions = element.getAttribute("data-rolloveractions");
if(actions) {
eval(actions);
}
event.stopPropagation();
event.preventDefault();
}
function onMouseOut(element, event) {
if (event.relatedTarget) {
if(isDescendantOf(event.relatedTarget, element)) return;
}
removeClass(element, '_idGenStateHover');
removeClass(element, '_idGenStateClick');
var actions = element.getAttribute("data-rolloffactions");
if(actions) {
eval(actions);
}
event.stopPropagation();
event.preventDefault();
}
function onTouchStart(element, event) {
if (hasAppearance(element, '_idGen-Appearance-Click')) {
addClass(element, '_idGenStateClick');
}
var actions = element.getAttribute("data-clickactions");
if(actions) {
eval(actions);
event.target.__id_touched = true;
}
event.stopPropagation();
event.preventDefault();
}
function onTouchEnd(element, event) {
removeClass(element, '_idGenStateClick');
var actions = element.getAttribute("data-releaseactions");
if(actions) {
eval(actions);
event.target.__id_touched = true;
}
event.stopPropagation();
event.preventDefault();
}
function onHide(id) {
var element=document.getElementById(id);
if(element) {
handleMSOStateParentOfObject(element);
addClass(element, '_idGenStateHide');
}
}
function handleMSOStateParentOfObject(element) {
var prev = element;
var parent = prev.parentNode;
var found;
while(parent && !found) {
var cn = parent.className;
if(cn && cn.indexOf('_idGenMSO') != -1)
found = true;
else
prev = parent;
parent = prev.parentNode;
}
if(found) {
var nextState = prev;
var mso_states = parent.children;
for (var i = 0, state; state = mso_states[i]; i++) {
var cn = state.className;
if (cn.indexOf('_idGenCurrentState') != -1 ) {
handleMediaInMSOState(state);
removeClass(state, '_idGenCurrentState');
addClass(state, '_idGenStateHide');
removeClass(nextState, '_idGenStateHide');
addClass(nextState, '_idGenCurrentState');
return;
}
}
}
}
function handleMediaInMSOState(element) {
/*This function is used to stop playing media present in current state when we move from current state to another state.*/
var descendants = element.getElementsByTagName('*');
for(var i = 0; i < descendants.length; i++) {
var e = descendants[i];
var tagName = e.tagName.toLowerCase();
if(tagName == 'video' || tagName == 'audio') {
if(!(e.paused)) {
e.currentTime = 0;
e.pause();
}
}
}
}
function playAnimatedElement(animated_element, className, hideAfterAnimating) {
removeClass(animated_element, '_idGenStateHide');
removeClass(animated_element, '_idGenPauseAnimation');
var cn = animated_element.className;
var previousAnimationClass = animated_element.getAttribute("data-idGenAnimationClass");
if ((cn.indexOf(className) == -1) && (cn.indexOf(previousAnimationClass) == -1)) {
addClass(animated_element, className);
animated_element.setAttribute("data-idGenAnimationClass", className);
}
else {
removeClass(animated_element, className);
removeClass(animated_element, previousAnimationClass);
animated_element.removeEventListener("webkitAnimationEnd", function(evt) { onPlayAnimationEnd(this, hideAfterAnimating, evt) });
animated_element.removeEventListener("animationend", function(evt) { onPlayAnimationEnd(this, hideAfterAnimating, evt) });
setTimeout(function() {addClass(animated_element, className)}, 10);
animated_element.setAttribute("data-idGenAnimationClass", className);
}
animated_element.addEventListener("webkitAnimationEnd", function(evt) { onPlayAnimationEnd(this, hideAfterAnimating, evt) });
animated_element.addEventListener("animationend", function(evt) { onPlayAnimationEnd(this, hideAfterAnimating, evt) });
}
function playAnimation(animation_id, className, startDelay, hideAfterAnimating) {
var animated_element = document.getElementById(animation_id);
if(animated_element) {
handleMSOStateParentOfObject(animated_element);
var startClassName = animated_element.getAttribute("data-idGenAnimationStartState");
var endClassName = animated_element.getAttribute("data-idGenAnimationEndState");
removeClass(animated_element, endClassName);
addClass(animated_element, startClassName);
setTimeout(function(){playAnimatedElement(animated_element, className, hideAfterAnimating)}, startDelay*1000);
}
}
function onPlayAnimationEnd(element, hideAfterAnimating, evt) {
var className = element.getAttribute("data-idGenAnimationClass");
var startClassName = element.getAttribute("data-idGenAnimationStartState");
var endClassName = element.getAttribute("data-idGenAnimationEndState");
removeClass(element, className);
removeClass(element, startClassName);
addClass(element, endClassName);
if(hideAfterAnimating)
addClass(element, '_idGenStateHide');
evt.stopPropagation();
}
function reverseAnimatedElement(animated_element, className) {
removeClass(animated_element, '_idGenStateHide');
removeClass(animated_element, '_idGenPauseAnimation');
var cn = animated_element.className;
var previousAnimationClass = animated_element.getAttribute("data-idGenAnimationClass");
if ((cn.indexOf(className) == -1) && (cn.indexOf(previousAnimationClass) == -1)) {
addClass(animated_element, className);
animated_element.setAttribute("data-idGenAnimationClass", className);
}
else {
removeClass(animated_element, className);
removeClass(animated_element, previousAnimationClass);
animated_element.removeEventListener("webkitAnimationEnd", function() { onReverseAnimationEnd(this) });
animated_element.removeEventListener("animationend", function() { onReverseAnimationEnd(this) });
setTimeout(function() {addClass(animated_element, className)}, 10);
animated_element.setAttribute("data-idGenAnimationClass", className);
}
animated_element.addEventListener("webkitAnimationEnd", function() { onReverseAnimationEnd(this) });
animated_element.addEventListener("animationend", function() { onReverseAnimationEnd(this) });
}
function reverseAnimation(animation_id, className, startDelay) {
var animated_element = document.getElementById(animation_id);
if(animated_element) {
handleMSOStateParentOfObject(animated_element);
var startClassName = animated_element.getAttribute("data-idGenAnimationStartState");
var endClassName = animated_element.getAttribute("data-idGenAnimationEndState");
removeClass(animated_element, startClassName);
addClass(animated_element, endClassName);
setTimeout(function(){reverseAnimatedElement(animated_element, className)}, startDelay*1000);
}
}
function onReverseAnimationEnd(element) {
var className = element.getAttribute("data-idGenAnimationClass");
var startClassName = element.getAttribute("data-idGenAnimationStartState");
var endClassName = element.getAttribute("data-idGenAnimationEndState");
removeClass(element, className);
removeClass(element, endClassName);
addClass(element, startClassName);
}
